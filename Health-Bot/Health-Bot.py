import discord
from discord import Embed
from discord.ext import commands
import numpy as np
import asyncio
from model import NeuralNetwork, input_size, hidden_layers, hidden_units, output_size, training_data, rewards, diagnosis_descriptions, augmented_training_data, augmented_rewards

# Initialize the neural network
network = NeuralNetwork(input_size, hidden_layers, hidden_units, output_size)

async def continuous_training(network, training_data, rewards, epochs=1, interval=10):
    while True:
        network.train(training_data, epochs, rewards)
        await asyncio.sleep(interval)

# Start continuous training
asyncio.run(continuous_training(network, augmented_training_data, augmented_rewards, epochs=10, interval=60))

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
                    value="This AI model has been trained on a limited dataset and may not diagnose you correctly. Use it as a guide and rely on your own judgment.",
                    inline=False)

    await ctx.send(embed=embed)


TOKEN = "token"

bot.run(TOKEN)
