import asyncio
import os

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

import oasis
from oasis import ActionType, EnvAction, SingleAction
os.environ["AZURE_API_VERSION"] = "2024-02-15-preview"
os.environ["AZURE_DEPLOYMENT_NAME"] = "gpt-4.1"


async def main():
  # Define the model for the agents
  azure_openai_model = ModelFactory.create(
      model_platform=ModelPlatformType.AZURE,
      model_type=ModelType.O1,
      api_key="8EQShlUHENwIgzAwUNnrM5ipuVUUaRaOVyPagaprwEbnSj0Q5A0oJQQJ99BEACHYHv6XJ3w3AAAAACOGA6A3",
      url="https://wyh17-max9q3c4-eastus2.cognitiveservices.azure.com/",
    )

  openai_model = ModelFactory.create(
      model_platform=ModelPlatformType.OPENAI,
      model_type=ModelType.GPT_4O_MINI,
      api_key="sk-pW0E44EugwGbsbxjY10Flcm89UiZI4BTgH02c3E1qvqPRsEl",
      url="https://www.dmxapi.com/v1",
  )
  
  # Define the available actions for the agents
  available_actions = [
      ActionType.LIKE_POST,
      ActionType.CREATE_POST,
      ActionType.CREATE_COMMENT,
      ActionType.FOLLOW
  ]

  # Make the environment
  env = oasis.make(
      platform=oasis.DefaultPlatformType.REDDIT,
      database_path="reddit_simulation.db",
      agent_profile_path="./data/reddit/user_data_36.json",
      agent_models=openai_model,
      available_actions=available_actions,
  )

  # Run the environment
  await env.reset()

  action = SingleAction(
    agent_id=0,
    action=ActionType.CREATE_POST,
    args={"content": "Welcome to the OASIS World!"}
  )

  env_actions = EnvAction(
    activate_agents=list(range(36)),  # activate the first 30 agents
    intervention=[action]
  )

  # Apply interventions to the environment, refresh the recommendation system, and LLM agent perform actions
  await env.step(env_actions)

  # Close the environment
  await env.close()

if __name__ == "__main__":
  asyncio.run(main())