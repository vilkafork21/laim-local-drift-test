"""
Конфигурация приложения.

Загружает настройки из переменных окружения и предоставляет конфигурации
для подключения к различным контурам (sigma/sds).
"""

import os

from dotenv import load_dotenv


# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================

class Config:
    def __init__(self):
        load_dotenv()
        credentials = os.environ.get("CREDENTIALS")
        auth_url = os.environ.get("AUTH_URL")
        base_url = os.environ.get("BASE_URL")
        scope = os.environ.get("SCOPE")
        verify_ssl_certs = (
            True if os.environ.get("VERIFY_SSL_CERTS") == "True" else False
        )

        AI_GATEWAY_URL = os.environ.get("AI_GATEWAY_URL", None)
        if AI_GATEWAY_URL is not None:
            self.contour = "sds"
            base_url = AI_GATEWAY_URL + "/api/v1"
        else:
            self.contour = "sigma"

        self.shared_config = {
            "base_url": base_url,
        }
        self.contour_configs = {
            "sigma": {
                "auth_url": auth_url,
                "credentials": credentials,
                "verify_ssl_certs": verify_ssl_certs,
                "scope": scope,
            }
            | self.shared_config,
            "sds": self.shared_config,
        }[self.contour]
