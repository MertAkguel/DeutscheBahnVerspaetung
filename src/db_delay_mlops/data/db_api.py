"""Minimal client for the DB API Marketplace (Timetables and StaDa).

The approach (raw responses, rate limiting, retries) follows
piebro/deutsche-bahn-data (MIT): https://github.com/piebro/deutsche-bahn-data
We use a synchronous client: with 60 calls/min and ~140 stations, concurrency
would not make collection faster, the rate limit is the bottleneck.
"""

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Self

import httpx
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

Api = Literal["timetables", "stada"]
RETRY_STATUS = {429, 500, 502, 503, 504}


class DBApiSettings(BaseSettings):
    """Read DB_CLIENT_ID and DB_API_KEY from the environment or from .env."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="DB_", extra="ignore")

    client_id: SecretStr
    api_key: SecretStr
    base_url: str = "https://apis.deutschebahn.com/db-api-marketplace/apis"
    # Limits of the Free plan
    timetables_calls_per_minute: int = 60
    stada_calls_per_second: int = 10


class RateLimiter:
    """Allow at most `calls` calls per `period` seconds by spacing them evenly."""

    def __init__(self, calls: int, period: float) -> None:
        self.interval = period / calls
        self._next_slot = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        if now < self._next_slot:
            time.sleep(self._next_slot - now)
        self._next_slot = max(now, self._next_slot) + self.interval


@dataclass(frozen=True)
class RawResponse:
    """One API call, stored as it came back. Failed calls are kept too, so gaps stay visible."""

    fetched_at: datetime
    endpoint: str
    eva: str | None
    url: str
    status_code: int | None
    body: str | None
    error: str | None
    attempts: int


class DBApiClient:
    def __init__(self, settings: DBApiSettings, max_retries: int = 5, timeout: float = 30) -> None:
        self.max_retries = max_retries
        self._http = httpx.Client(
            base_url=settings.base_url,
            headers={
                "DB-Client-Id": settings.client_id.get_secret_value(),
                "DB-Api-Key": settings.api_key.get_secret_value(),
            },
            timeout=timeout,
        )
        self._limiters: dict[Api, RateLimiter] = {
            "timetables": RateLimiter(settings.timetables_calls_per_minute, 60),
            "stada": RateLimiter(settings.stada_calls_per_second, 1),
        }

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self._http.close()

    def plan(self, eva: str, hour: datetime) -> RawResponse:
        """Planned timetable of one station for one hour (local German time)."""
        return self._get(f"/timetables/v1/plan/{eva}/{hour:%y%m%d}/{hour:%H}", api="timetables", endpoint="plan", eva=eva)

    def fchg(self, eva: str) -> RawResponse:
        """All currently known changes (delays, cancellations) of one station."""
        return self._get(f"/timetables/v1/fchg/{eva}", api="timetables", endpoint="fchg", eva=eva)

    def station(self, eva: str) -> RawResponse:
        """StaDa master data (name, federal state, coordinates) of the station with this EVA number."""
        return self._get(
            "/station-data/v2/stations",
            api="stada",
            endpoint="stada",
            eva=eva,
            params={"eva": eva.lstrip("0")},
            accept="application/json",
        )

    def _get(
        self,
        path: str,
        *,
        api: Api,
        endpoint: str,
        eva: str | None,
        params: dict[str, str] | None = None,
        accept: str = "application/xml",
    ) -> RawResponse:
        url = str(self._http.build_request("GET", path, params=params).url)
        status_code: int | None = None
        error: str | None = None

        for attempt in range(1, self.max_retries + 1):
            self._limiters[api].wait()
            try:
                response = self._http.get(path, params=params, headers={"Accept": accept})
            except httpx.TransportError as exc:  # timeouts, connection errors
                status_code, error = None, repr(exc)
            else:
                status_code = response.status_code
                if status_code not in RETRY_STATUS:
                    return RawResponse(
                        fetched_at=datetime.now(UTC),
                        endpoint=endpoint,
                        eva=eva,
                        url=url,
                        status_code=status_code,
                        body=response.text,
                        error=None if response.is_success else f"HTTP {status_code}",
                        attempts=attempt,
                    )
                error = f"HTTP {status_code}"

            if attempt < self.max_retries:
                backoff = min(2**attempt, 60)  # 2, 4, 8, 16 s
                logger.warning("%s failed (%s), retry %d in %ds", url, error, attempt, backoff)
                time.sleep(backoff)

        logger.error("%s failed after %d attempts: %s", url, self.max_retries, error)
        return RawResponse(datetime.now(UTC), endpoint, eva, url, status_code, None, error, self.max_retries)
