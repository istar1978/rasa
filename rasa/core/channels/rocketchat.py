import logging
from sanic import Blueprint, response
from sanic.request import Request
from typing import Text, Dict, Any, List, Iterable, Optional, Callable, Awaitable

from rasa.core.channels.channel import UserMessage, OutputChannel, InputChannel
from sanic.response import HTTPResponse

logger = logging.getLogger(__name__)


class RocketChatBot(OutputChannel):
    @classmethod
    def name(cls) -> Text:
        return "rocketchat"

    def __init__(self, user, password, server_url):
        from rocketchat_API.rocketchat import RocketChat

        self.rocket = RocketChat(user, password, server_url=server_url)

    async def send_response(self, recipient_id: Text, message: Dict[Text, Any]) -> None:
        """Send a message to the user."""

        response = {
            "text": "",
            "room_id": recipient_id,
        }
        if message.get("custom"):
            self.customize_response(response, message.get("custom"))
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
            if message.get("elements"):
                self.add_attachments(response, message.get("elements"))
            if message.get("attachment"):
                self.add_attachments(response, [message.get("attachment")])

        await self.rocket.__call_api_post("chat.postMessage", **response)

    @staticmethod
    def add_text(response: Dict[Text, Any], text: Text) -> None:
        response.update({"text": text})

    @staticmethod
    def add_image(response: Dict[Text, Any], image_url: Text) -> None:
        attachments = response.get("attachments", [])
        attachments.append({"image_url": image_url, "collapsed": False})

        response.update({"attachments": attachments})

    @staticmethod
    def add_attachments(
        response: Dict[Text, Any], attachments: Iterable[Dict[Text, Any]]
    ) -> None:
        response_attachments = response.get("attachments", [])
        for attachment in attachments:
            response_attachments.append(attachment)

        response.update({"attachments": response_attachments})

    @staticmethod
    def add_text_with_buttons(
        response: Dict[Text, Any], text: Text, buttons: List[Dict[Text, Any]],
    ) -> None:
        button_block = {"actions": []}
        for button in buttons:
            button_block["actions"].append(
                {
                    "text": button["title"],
                    "msg": button["payload"],
                    "type": "button",
                    "msg_in_chat_window": True,
                }
            )
        attachments = response.get("attachments", [])
        attachments.append(button_block)
        response.update({"text": text, "attachments": attachments})

    @staticmethod
    def customize_response(
        response: Dict[Text, Any], json_message: Dict[Text, Any]
    ) -> None:

        remove_room_id = False
        if json_message.get("channel"):
            if json_message.get("room_id"):
                logger.warning(
                    "Only one of `channel` or `room_id` can be passed to a RocketChat "
                    "message post. Defaulting to `channel`."
                )
                remove_room_id = True

        response.update(json_message)
        if remove_room_id:
            del response["room_id"]


class RocketChatInput(InputChannel):
    """RocketChat input channel implementation."""

    @classmethod
    def name(cls) -> Text:
        return "rocketchat"

    @classmethod
    def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> InputChannel:
        if not credentials:
            cls.raise_missing_credentials_exception()

        # pytype: disable=attribute-error
        return cls(
            credentials.get("user"),
            credentials.get("password"),
            credentials.get("server_url"),
        )
        # pytype: enable=attribute-error

    def __init__(self, user: Text, password: Text, server_url: Text) -> None:

        self.user = user
        self.password = password
        self.server_url = server_url

    async def send_message(
        self,
        text: Optional[Text],
        sender_name: Optional[Text],
        recipient_id: Optional[Text],
        on_new_message: Callable[[UserMessage], Awaitable[Any]],
        metadata: Optional[Dict],
    ):
        if sender_name != self.user:
            output_channel = self.get_output_channel()

            user_msg = UserMessage(
                text,
                output_channel,
                recipient_id,
                input_channel=self.name(),
                metadata=metadata,
            )
            await on_new_message(user_msg)

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[Any]]
    ) -> Blueprint:
        rocketchat_webhook = Blueprint("rocketchat_webhook", __name__)

        @rocketchat_webhook.route("/", methods=["GET"])
        async def health(_: Request) -> HTTPResponse:
            return response.json({"status": "ok"})

        @rocketchat_webhook.route("/webhook", methods=["GET", "POST"])
        async def webhook(request: Request) -> HTTPResponse:
            output = request.json
            metadata = self.get_metadata(request)
            if output:
                if "visitor" not in output:
                    sender_name = output.get("user_name", None)
                    text = output.get("text", None)
                    recipient_id = output.get("channel_id", None)
                else:
                    messages_list = output.get("messages", None)
                    text = messages_list[0].get("msg", None)
                    sender_name = messages_list[0].get("username", None)
                    recipient_id = output.get("_id")

                await self.send_message(
                    text, sender_name, recipient_id, on_new_message, metadata
                )

            return response.text("")

        return rocketchat_webhook

    def get_output_channel(self) -> OutputChannel:
        return RocketChatBot(self.user, self.password, self.server_url)
