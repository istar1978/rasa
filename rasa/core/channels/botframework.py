# -*- coding: utf-8 -*-

import datetime
import json
import logging
import requests
from sanic import Blueprint, response
from sanic.request import Request
from typing import Text, Dict, Any, List, Iterable, Callable, Awaitable, Optional

from rasa.core.channels.channel import UserMessage, OutputChannel, InputChannel
from sanic.response import HTTPResponse

logger = logging.getLogger(__name__)

MICROSOFT_OAUTH2_URL = "https://login.microsoftonline.com"

MICROSOFT_OAUTH2_PATH = "botframework.com/oauth2/v2.0/token"


class BotFramework(OutputChannel):
    """A Microsoft Bot Framework communication channel."""

    token_expiration_date = datetime.datetime.now()

    headers = None

    @classmethod
    def name(cls) -> Text:
        return "botframework"

    def __init__(
        self,
        app_id: Text,
        app_password: Text,
        conversation: Dict[Text, Any],
        bot: Text,
        service_url: Text,
    ) -> None:

        self.app_id = app_id
        self.app_password = app_password
        self.conversation = conversation
        self.global_uri = "{}v3/".format(service_url)
        self.bot = bot

    async def _get_headers(self):
        if BotFramework.token_expiration_date < datetime.datetime.now():
            uri = "{}/{}".format(MICROSOFT_OAUTH2_URL, MICROSOFT_OAUTH2_PATH)
            grant_type = "client_credentials"
            scope = "https://api.botframework.com/.default"
            payload = {
                "client_id": self.app_id,
                "client_secret": self.app_password,
                "grant_type": grant_type,
                "scope": scope,
            }

            token_response = requests.post(uri, data=payload)

            if token_response.ok:
                token_data = token_response.json()
                access_token = token_data["access_token"]
                token_expiration = token_data["expires_in"]

                delta = datetime.timedelta(seconds=int(token_expiration))
                BotFramework.token_expiration_date = datetime.datetime.now() + delta

                BotFramework.headers = {
                    "content-type": "application/json",
                    "Authorization": "Bearer %s" % access_token,
                }
                return BotFramework.headers
            else:
                logger.error("Could not get BotFramework token")
        else:
            return BotFramework.headers

    async def send(self, message_data: Dict[Text, Any]) -> None:
        post_message_uri = "{}conversations/{}/activities".format(
            self.global_uri, self.conversation["id"]
        )
        headers = await self._get_headers()
        send_response = requests.post(
            post_message_uri, headers=headers, data=json.dumps(message_data)
        )

        if not send_response.ok:
            logger.error(
                f"Error trying to send botframework message. Response: {send_response.text}",
            )

    async def send_response(self, recipient_id: Text, message: Dict[Text, Any]) -> None:
        """Send a message to the user."""

        response = {
            "type": "message",
            "recipient": {"id": recipient_id},
            "from": self.bot,
            "channelData": {"notification": {"alert": "true"}},
            "text": "",
        }
        if message.get("custom"):
            self.add_custom_json(response, message.get("custom"))
        elif message.get("buttons"):
            self.add_text_with_buttons(
                response, message.get("text", ""), message.get("buttons")
            )
        else:
            if message.get("text"):
                # TODO: handle multiple messages with \n\n?
                self.add_text(response, message.get("text"))
            if message.get("image"):
                self.add_image(response, message.get("image"))

        await self.send(response)

    @staticmethod
    def add_text(response: Dict[Text, Any], text: Text) -> None:
        response.update({"text": text})

    @staticmethod
    def add_image(response: Dict[Text, Any], image_url: Text) -> None:
        hero_content = {
            "contentType": "application/vnd.microsoft.card.hero",
            "content": {"images": [{"url": image_url}]},
        }
        attachments = response.get("attachments", [])
        attachments.append(hero_content)
        response.update({"attachments": attachments})

    @staticmethod
    def add_text_with_buttons(
        response: Dict[Text, Any], text: Text, buttons: List[Dict[Text, Any]],
    ) -> None:
        hero_content = {
            "contentType": "application/vnd.microsoft.card.hero",
            "content": {"subtitle": text, "buttons": buttons,},
        }
        response.update({"attachments": [hero_content]})

    @staticmethod
    def add_custom_json(
        response: Dict[Text, Any], json_message: Dict[Text, Any]
    ) -> None:
        response.update(json_message)


class BotFrameworkInput(InputChannel):
    """Bot Framework input channel implementation."""

    @classmethod
    def name(cls) -> Text:
        return "botframework"

    @classmethod
    def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> InputChannel:
        if not credentials:
            cls.raise_missing_credentials_exception()

        # pytype: disable=attribute-error
        return cls(credentials.get("app_id"), credentials.get("app_password"))
        # pytype: enable=attribute-error

    def __init__(self, app_id: Text, app_password: Text) -> None:
        """Create a Bot Framework input channel.

        Args:
            app_id: Bot Framework's API id
            app_password: Bot Framework application secret
        """

        self.app_id = app_id
        self.app_password = app_password

    @staticmethod
    def add_attachments_to_metadata(
        postdata: Dict[Text, Any], metadata: Optional[Dict[Text, Any]]
    ) -> Optional[Dict[Text, Any]]:
        """Merge the values of `postdata['attachments']` with `metadata`."""

        if postdata.get("attachments"):
            attachments = {"attachments": postdata["attachments"]}
            if metadata:
                metadata.update(attachments)
            else:
                metadata = attachments

        return metadata

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[Any]]
    ) -> Blueprint:

        botframework_webhook = Blueprint("botframework_webhook", __name__)

        # noinspection PyUnusedLocal
        @botframework_webhook.route("/", methods=["GET"])
        async def health(request: Request) -> HTTPResponse:
            return response.json({"status": "ok"})

        @botframework_webhook.route("/webhook", methods=["POST"])
        async def webhook(request: Request) -> HTTPResponse:
            postdata = request.json
            metadata = self.get_metadata(request)

            metadata_with_attachments = self.add_attachments_to_metadata(
                postdata, metadata
            )

            try:
                if postdata["type"] == "message":
                    out_channel = BotFramework(
                        self.app_id,
                        self.app_password,
                        postdata["conversation"],
                        postdata["recipient"],
                        postdata["serviceUrl"],
                    )

                    user_msg = UserMessage(
                        text=postdata.get("text", ""),
                        output_channel=out_channel,
                        sender_id=postdata["from"]["id"],
                        input_channel=self.name(),
                        metadata=metadata_with_attachments,
                    )

                    await on_new_message(user_msg)
                else:
                    logger.info("Not received message type")
            except Exception as e:
                logger.error("Exception when trying to handle message.{0}".format(e))
                logger.debug(e, exc_info=True)
                pass

            return response.text("success")

        return botframework_webhook
