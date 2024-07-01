import random
import json
import torch
from kivy.uix.boxlayout import BoxLayout
from kivymd.app import MDApp
from kivymd.uix.textfield import MDTextField
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.graphics import Color, Rectangle
from kivy.metrics import dp

from model import NeuralNet
from preprocessing import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ChatApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title = "ChatBot"
        self.bot_name = "NEU"
        self.history = []


        with open('intents.json', 'r') as json_data:
            self.intents = json.load(json_data)

        FILE = "data.pth"
        data = torch.load(FILE)

        input_size = data["input_size"]
        hidden_size = data["hidden_size"]
        output_size = data["output_size"]
        self.all_words = data['all_words']
        self.tags = data['tags']
        model_state = data["model_state"]

        self.model = NeuralNet(input_size, hidden_size, output_size).to(device)
        self.model.load_state_dict(model_state)
        self.model.eval()

    def build(self):
        layout = BoxLayout(orientation='vertical')

        # Create a box layout for the bot name with a background color
        bot_name_layout = BoxLayout(size_hint_y=None, height=60)
        with bot_name_layout.canvas.before:
            Color(0.9, 0.9, 0.9, 1)  # Light gray background color
            self.rect = Rectangle(size=bot_name_layout.size, pos=bot_name_layout.pos)
            bot_name_layout.bind(size=self._update_rect, pos=self._update_rect)

        bot_name_label = Label(
            text=f"{self.bot_name}",
            font_size='36sp',  # Increase font size
            color=(1, 0, 0, 1)
        )
        bot_name_layout.add_widget(bot_name_label)
        layout.add_widget(bot_name_layout)

        # Scrollable chat history
        self.scroll_view = ScrollView(size_hint=(1, 1))
        self.history_layout = BoxLayout(orientation='vertical', size_hint_y=None, padding=dp(10))
        self.history_layout.bind(minimum_height=self.history_layout.setter('height'))
        self.scroll_view.add_widget(self.history_layout)
        layout.add_widget(self.scroll_view)

        # Input field
        self.input_text = MDTextField(
            hint_text="Type here...",
            size_hint=(1, None),
            height=30,
            multiline=False
        )
        self.input_text.bind(on_text_validate=self.send_message)
        layout.add_widget(self.input_text)

        # Send button
        send_button = Button(text="Send", size_hint=(1, None), height=40, on_release=self.send_message)
        layout.add_widget(send_button)

        return layout

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def send_message(self, instance):
        user_input = self.input_text.text
        self.input_text.text = ""  # Clear input field
        if user_input.strip():  # Check if user input is not empty
            self.update_chat_history(f"You: {user_input}", 'user')
            bot_response = self.get_bot_response(user_input)
            self.update_chat_history(f"{self.bot_name}: {bot_response}", 'bot')

    def update_chat_history(self, message, sender):
        if sender == 'user':
            color = (1, 0, 0, 1)  # Red for "You:"
        elif sender == 'bot':
            color = (0, 0, 1, 1)  # Blue for "Sam:"
        else:
            color = (0, 0, 0, 1)  # Black for other text

        message_label = Label(
            text=message,
            size_hint_y=None,
            color=color,
            text_size=(self.history_layout.width - dp(20), None),
            halign='left',
            valign='top',
            padding=(dp(10), dp(10))
        )
        message_label.bind(texture_size=message_label.setter('size'))
        self.history_layout.add_widget(message_label)
        # Scroll to the bottom
        self.scroll_view.scroll_y = 0

    def get_bot_response(self, user_input):
        # Tokenize user input
        sentence = tokenize(user_input)
        X = bag_of_words(sentence, self.all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        # Get model output
        output = self.model(X)
        _, predicted = torch.max(output, dim=1)
        tag = self.tags[predicted.item()]

        # Check if predicted tag matches any intent
        for intent in self.intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                return response

        return "I do not understand..."

if __name__ == "__main__":
    ChatApp().run()
