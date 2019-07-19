## Supervised Response Selector Component

This experimental branch introduces a new feature component which we internally 
call as the **Supervised Response Selector**. We envision this component to come in handy
in cases where your bot has to deal with open domain questions like - 

1. "what's the weather now?" 
2. "who built you?"
3. "what's your name?"
4. "what can you do for me?"

With the current stack, utterances similar to these questions should be given an intent and
appropriate multiple actions should be defined to answer them coherently. Since these utterances
can be non-contributing to user goal, including them in your Rasa story can be difficult too.

We propose to group such utterances with their possible responses under one large intent called **chitchat**
and train a ML model to select the correct response given the utterance.

The feature is implemented as a custom component called `ResponseSelector` within NLU pipeline and works
very similarly to `EmbeddingIntentClassifier` component.


### How to Start

#### Code

The current codebase of this feature is being pushed to `supervised_response_selector` 
[branch](https://github.com/RasaHQ/rasa/tree/supervised_response_selector) of Rasa repo. Clone it and
install it from source.

We also integrated(bleeding edge) the feature with our own demo-bot Sara on an experimental 
[branch](https://github.com/RasaHQ/rasa-demo/tree/supervised_response_selector).

#### Training Data

Currently, only markdown files are supported as training data for this feature.
The training data format is kept simple and very similar to intent classification inside NLU. Here is an 
example -

```
## response:I_dont_know_about_where_you_live_but_in_my_world_its_always_sunny
- what the weather like?
- weather at you location?
- weather be like at your place?
- what is the weather in Berlin
- what's the weather like
- hows the weather today in berlin?
- whats the temperature
- whats the temperature in delhi?
- what is the weather like?
- the weather today
- What is the weather in newyork?
- How is weather today
```

So, instead of `intent` as the heading, use `response` to indicate that `ResponseSelector` component has
to pick up this as a training example. These training examples can even be a part of the training data file which
consists examples for Intent Classifier - 

```
## intent:signup_newsletter
- I wanna sign up for the newsletter.
- I want to sign up for the newsletter.
- I would like to sign up for the newsletter.
- Sign me up for the newsletter.
- Sign up.
- Newsletter please.

## response:I_dont_know_about_where_you_live_but_in_my_world_its_always_sunny
- what the weather like?
- weather at you location?
- weather be like at your place?
- what is the weather in Berlin
- what's the weather like
```

Each training example which is tagged with `response` heading automatically gets added to set of training examples of 
intent classifier as well with a default intent `chitchat`


#### NLU Config

Add `ResponseSelector` component to NLU pipeline - 

```
- name: ResponseSelector
  epochs: 40
  evaluate_every_num_epochs: 5
  evaluate_on_num_examples: 256
  hidden_layers_sizes_a:
    - 128
    - 64
  hidden_layers_sizes_b:
    - 128
    - 64
```

Take a look at the example [config](https://github.com/RasaHQ/rasa-demo/blob/supervised_response_selector/config.yml)
 from demo bot repo.
 
 Also, if you are including the `ResponseSelector` component, don't forget to include the intent `chitchat`
 in your `domain.yml` [file](https://github.com/RasaHQ/rasa-demo/blob/supervised_response_selector/domain.yml#L28).



#### Custom Action

Write a custom action which basically picks up the response predicted by `ResponseSelector` and utters it as
a message.

```python
class ActionChitchat(Action):
    """Returns the chitchat utterance dependent on the intent"""

    def name(self):
        return "action_chitchat"

    def run(self, dispatcher, tracker, domain):

        response = tracker.latest_message["response"].get("name")
        dispatcher.utter_message(' '.join(response.split('_')))
        return []

```

#### Stories

Write your Rasa stories which include handling chitchat utterances and triggering `action_chitchat` 
appropriately. Take a look at this [file](https://github.com/RasaHQ/rasa-demo/blob/supervised_response_selector/data/core/chitchat.md)
for such examples.

----------------------------------
