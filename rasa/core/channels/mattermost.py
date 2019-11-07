import logging
from mattermostwrapper import MattermostAPI
from sanic import Blueprint, response
from sanic.request import Request
from typing import Text, Dict, Any, List, Callable, Awaitable, Optional

from rasa.core.channels.channel import UserMessage, OutputChannel, InputChannel
from sanic.response import HTTPResponse

logger = logging.getLogger(__name__)


class MattermostBot(MattermostAPI, OutputChannel):
    """A Mattermost communication channel"""

    @classmethod
    def name(cls) -> Text:
        return "mattermost"

    @classmethod
    def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> OutputChannel:
        if not credentials:
            cls.raise_missing_credentials_exception()

        return cls(credentials.get("webhook_url"))  # pytype: disable=attribute-error

    def __init__(
        self,
        url: Text,
        team: Text,
        user: Text,
        pw: Text,
        bot_channel: Text,
        webhook_url: Optional[Text],
    ) -> None:
        self.url = url
        self.team = team
        self.user = user
        self.pw = pw
        self.bot_channel = bot_channel
        self.webhook_url = webhook_url

        super(MattermostBot, self).__init__(url, team)
        super(MattermostBot, self).login(user, pw)

    async def send_response(self, recipient_id: Text, message: Dict[Text, Any]) -> None:
        """Send a message to the user."""

        response = {
            "channel_id": self.bot_channel,
            "message": "",
            "props": {},
        }
        if message.get("custom"):
            response.update(message.get("custom"))
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

        await self.post("/posts", response)

    @staticmethod
    def add_text(response: Dict[Text, Any], text: Text) -> None:
        response.update({"message": text})

    @staticmethod
    def add_image(response: Dict[Text, Any], image_url: Text) -> None:
        props = response.get("props")
        props.setdefault({"attachments": []})
        props["attachments"].append({"image_url": image_url})

        response.update({"props": props})

    def add_text_with_buttons(
        self, response: Dict[Text, Any], text: Text, buttons: List[Dict[Text, Any]],
    ) -> None:
        button_block = {"actions": []}
        for button in buttons:
            button_block["actions"].append(
                {
                    "name": button["title"],
                    "integration": {
                        "url": self.webhook_url,
                        "context": {"action": button["payload"]},
                    },
                }
            )
        props = response.get("props")
        props.setdefault({"attachments": []})
        props["attachments"].append(button_block)

        response.update({"message": text, "props": props})


class MattermostInput(InputChannel):
    """Mattermost input channel implemenation."""

    @classmethod
    def name(cls) -> Text:
        return "mattermost"

    @classmethod
    def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> InputChannel:
        if not credentials:
            cls.raise_missing_credentials_exception()

        # pytype: disable=attribute-error
        return cls(
            credentials.get("url"),
            credentials.get("team"),
            credentials.get("user"),
            credentials.get("pw"),
            credentials.get("webhook_url"),
        )
        # pytype: enable=attribute-error

    def __init__(
        self, url: Text, team: Text, user: Text, pw: Text, webhook_url: Text
    ) -> None:
        """Create a Mattermost input channel.
        Needs a couple of settings to properly authenticate and validate
        messages.

        Args:
            url: Your Mattermost team url including /v4 example
                https://mysite.example.com/api/v4
            team: Your mattermost team name
            user: Your mattermost userid that will post messages
            pw: Your mattermost password for your user
            webhook_url: The mattermost callback url as specified
                in the outgoing webhooks in mattermost example
                https://mysite.example.com/webhooks/mattermost/webhook
        """
        self.url = url
        self.team = team
        self.user = user
        self.pw = pw
        self.webhook_url = webhook_url

    async def message_with_trigger_word(
        self,
        on_new_message: Callable[[UserMessage], Awaitable[None]],
        output: Dict[Text, Any],
        metadata: Optional[Dict],
    ) -> None:
        # splitting to get rid of the @botmention
        # trigger we are using for this
        text = output["text"].split(" ", 1)
        text = text[1]

        sender_id = output["user_id"]
        self.bot_channel = output["channel_id"]

        try:
            out_channel = MattermostBot(
                self.url,
                self.team,
                self.user,
                self.pw,
                self.bot_channel,
                self.webhook_url,
            )
            user_msg = UserMessage(
                text,
                out_channel,
                sender_id,
                input_channel=self.name(),
                metadata=metadata,
            )
            await on_new_message(user_msg)
        except Exception as e:
            logger.error("Exception when trying to handle message.{0}".format(e))
            logger.debug(e, exc_info=True)

    async def action_from_button(
        self,
        on_new_message: Callable[[UserMessage], Awaitable[None]],
        output: Dict[Text, Any],
        metadata: Optional[Dict],
    ) -> None:
        # get the action, the buttons triggers
        action = output["context"]["action"]

        sender_id = output["user_id"]
        self.bot_channel = output["channel_id"]

        try:
            out_channel = MattermostBot(
                self.url,
                self.team,
                self.user,
                self.pw,
                self.bot_channel,
                self.webhook_url,
            )
            context_action = UserMessage(
                action,
                out_channel,
                sender_id,
                input_channel=self.name(),
                metadata=metadata,
            )
            await on_new_message(context_action)
        except Exception as e:
            logger.error("Exception when trying to handle message.{0}".format(e))
            logger.debug(e, exc_info=True)

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[None]]
    ) -> Blueprint:
        mattermost_webhook = Blueprint("mattermost_webhook", __name__)

        @mattermost_webhook.route("/", methods=["GET"])
        async def health(_: Request) -> HTTPResponse:
            return response.json({"status": "ok"})

        @mattermost_webhook.route("/webhook", methods=["POST"])
        async def webhook(request: Request) -> HTTPResponse:
            output = request.json

            if not output:
                return response.text("")

            metadata = self.get_metadata(request)
            # handle normal message with trigger_word
            if "trigger_word" in output:
                await self.message_with_trigger_word(on_new_message, output, metadata)

            # handle context actions from buttons
            elif "context" in output:
                await self.action_from_button(on_new_message, output, metadata)

            return response.text("success")

        return mattermost_webhook
