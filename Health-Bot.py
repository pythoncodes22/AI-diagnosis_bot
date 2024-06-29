import discord
from discord import Embed
from discord.ext import commands
import numpy as np
import tensorflow as tf
import asyncio

input_size = 3 
hidden_layers = 8  # Number of hidden layers
hidden_units = 512  # Number of neurons in each hidden layer
output_size = 3  # Number of possible outcomes


diagnosis_descriptions = {
    1: "Diagnosis type 1: Go to the doctor.",
    2: "Diagnosis type 2: Don't go to the doctor, you should be fine.",
    3: "Diagnosis type 3: Go to the hospital.",
}

# Define the training data
training_data = [
    {"test": [5, 10, 9], "diagnosis": 3},  # Example 1: Game state [5, 10, 9], Diagnosis: 3
    {"test": [1, 1, 2], "diagnosis": 2},
    {"test": [3, 2, 6], "diagnosis": 1},
    {"test": [1, 1, 4], "diagnosis": 1},
    {"test": [2, 1, 1], "diagnosis": 2},
    # Add more training data examples as needed
]

# Define rewards associated with each training example
rewards = [1, 1, 1]  # Example rewards, adjust according to your application

# Data augmentation function
def augment_data(data, num_augmentations):
    augmented_data = []
    for example in data:
        for _ in range(num_augmentations):
            augmented_example = {
                "test": np.clip(np.array(example["test"]) + np.random.normal(0, 0.1, size=len(example["test"])), 0, 10),
                "diagnosis": example["diagnosis"]
            }
            augmented_data.append(augmented_example)
    return augmented_data

# Augment the training data
augmented_training_data = augment_data(training_data, num_augmentations=10)
augmented_rewards = rewards * 10
class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, hidden_units, output_size):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.output_size = output_size

        model_layers = [
            tf.keras.layers.Dense(hidden_units, input_shape=(input_size,), activation='relu'),
            tf.keras.layers.BatchNormalization(),  # Batch normalization layer
            tf.keras.layers.Dropout(0.5)  # Dropout layer with 50% dropout rate
        ]

        for _ in range(hidden_layers - 1):
            model_layers.append(tf.keras.layers.Dense(hidden_units, activation='relu'))
            model_layers.append(tf.keras.layers.BatchNormalization())  # Batch normalization layer
            model_layers.append(tf.keras.layers.Dropout(0.5))  # Dropout layer with 50% dropout rate

        model_layers.append(tf.keras.layers.Dense(output_size, activation='softmax'))

        self.model = tf.keras.Sequential(model_layers)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def predict(self, input_vector):
        return self.model.predict(input_vector)

    def train(self, training_data, epochs, rewards):
        input_data = np.array([example['test'] for example in training_data])
        target_data = np.array(
            [example['diagnosis'] - 1 for example in training_data])  # Adjust move indices to start from 0

        self.model.fit(input_data, target_data, epochs=epochs, sample_weight=np.array(rewards), verbose=0)


# Initialize the complex neural network
network = NeuralNetwork(input_size, hidden_layers, hidden_units, output_size)

# Define bot parameters
cmd_prefix = "!"
intents = discord.Intents.all()
bot = commands.Bot(command_prefix=cmd_prefix, intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')
    await bot.change_presence(activity=discord.Game(name=f"{cmd_prefix}info"))


@bot.command()
async def predict(ctx):
    def check(msg):
        return msg.author == ctx.author and msg.channel == ctx.channel

    embed = Embed(title="Health Test", description="Please enter your health state in the format 'Time period in hours, How you felt at the beginning of the time period on a scale of 1 to 10, How you felt at the end of the time period on a scale of 1 to 10'example: 5 10 9",
                  color=discord.Color.blue())
    await ctx.send(embed=embed)

    try:
        state_msg = await bot.wait_for('message', check=check, timeout=30)
        game_state = list(map(int, state_msg.content.split()))
        if len(game_state) != input_size:
            embed = Embed(title="Invalid test", description=f"The test should have {input_size} values.",
                          color=discord.Color.red())
            await ctx.send(embed=embed)
            return
        padded_game_state = game_state + [0] * (input_size - len(game_state))

    except asyncio.TimeoutError:
        embed = Embed(title="Timeout", description="Timed out. Please try again.", color=discord.Color.red())
        await ctx.send(embed=embed)
        return

    # Predict the next move based on the padded game state
    prediction = network.predict(np.array([padded_game_state]))
    predicted_move = np.argmax(prediction) + 1

    embed = Embed(title="Prediction", color=discord.Color.green())
    embed.add_field(name="diagnosis:", value=str(predicted_move))
    await ctx.send(embed=embed)




@bot.command()
async def support(ctx):
    embed = Embed(title="diagnosis types", color=discord.Color.blue())

    for diagnosis, description in diagnosis_descriptions.items():
        embed.add_field(name=f"type {diagnosis}", value=description, inline=False)

    await ctx.send(embed=embed)


@bot.command()
async def train(ctx, epochs: int):
    embed = Embed(title="Training", description="Training the AI model...", color=discord.Color.blue())
    message = await ctx.send(embed=embed)

    augmentation_factor = 1000  # Define the number of times data augmentation should be applied

    augmented_training_data = []
    augmented_rewards = []

    for example, reward in zip(training_data, rewards):
        augmented_training_data.append(example)
        augmented_rewards.append(reward)

        for _ in range(augmentation_factor):
            # Apply data augmentation by randomly modifying the game state
            augmented_state = np.array(example["test"]) + np.random.randint(-2, 3, size=input_size)
            augmented_state = np.clip(augmented_state, 0, 10)  # Clip values between 0 and 10
            augmented_training_data.append({"test": augmented_state.tolist(), "diagnosis": example["diagnosis"]})
            augmented_rewards.append(reward)

    try:
        await asyncio.sleep(0)  # Allow other tasks to run

        # Run the training process in the background
        await asyncio.get_event_loop().run_in_executor(None, network.train, augmented_training_data, epochs,
                                                       augmented_rewards)

        embed = Embed(title="Training Completed", description="Training completed.", color=discord.Color.green())
        await ctx.send(embed=embed)  # Send an embed indicating training is complete
    except Exception as e:
        embed = Embed(title="Training Error", description=f"An error occurred during training: {str(e)}",
                      color=discord.Color.red())
        await message.edit(embed=embed)
  # Edit the original message with the error message


# Edit the original message with the error message
# Send a message indicating training is complete


@bot.command()
async def feedback(ctx, *args):
    if len(args) != input_size + 1:
        embed = Embed(title="Invalid Feedback",
                      description="Invalid feedback. Please provide the test followed by the correct diagnosis.",
                      color=discord.Color.red())
        await ctx.send(embed=embed)
        return

    try:
        test = [int(arg) for arg in args[:-1]]
        diagnosis = int(args[-1])
    except ValueError:
        embed = Embed(title="Invalid Feedback",
                      description="Invalid feedback. Please provide valid integer values for the test and diagnosis.",
                      color=discord.Color.red())
        await ctx.send(embed=embed)
        return

    if diagnosis not in diagnosis_descriptions:
        embed = Embed(title="Invalid diagnosis", description="Invalid diagnosis. Please provide a valid diagnosis number.",
                      color=discord.Color.red())
        await ctx.send(embed=embed)
        return

    feedback_data = {
        "test": test,
        "diagnosis": diagnosis
    }

    # Add the feedback data to the training dataset
    training_data.append(feedback_data)

    embed = Embed(title="Feedback Received",
                  description="Thank you for your feedback. The AI model will be updated based on the provided information.",
                  color=discord.Color.green())
    await ctx.send(embed=embed)


@bot.command()
async def info(ctx):
    embed = Embed(title="Doctor AI", description="An AI-powered Doctor",
                  color=discord.Color.magenta())

    embed.add_field(name="Commands",
                    value="!predict - Get Diagnosed by the AI\n!support - diagnosis types descriptions\n!train <epochs> - Train the AI model\n!info - Display information about the AI\n!feedback <test and correct diagnosis> - Provide feedback for an incorrect diagnosis",
                    inline=False)
    embed.add_field(name="About",
                    value="This AI model has been trained on a limited dataset and may not diagnose you corectly. Use it as a guide and rely on your own judgment.",
                    inline=False)

    await ctx.send(embed=embed)


TOKEN = "MTExNzM3OTM1NjA1OTI1MDc0OQ.GcCSZ6.IIYalJZyrkp1xMftKLaUtmNV_kn5JfJ10kpXk0"

bot.run(TOKEN)