# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A factory implementing the three key questions agent as an entity."""

import datetime

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.utils import measurements as measurements_lib

DEFAULT_PLANNING_HORIZON = 'the rest of the day, focusing most on the near term'


def get_pomodoro_reminder(overarching_goal, agent_name):
  def pomodoro_reminder(chain_of_thought, clock_now):
    """Reminds the agent to stay on topic"""

    return f'{agent_name} should stay on topic: {overarching_goal}.'

  return pomodoro_reminder


def _get_class_name(object_: object) -> str:
  return object_.__class__.__name__


def build_agent(
    *,
    config: formative_memories.AgentConfig,
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemory,
    clock: game_clock.MultiIntervalClock,
    update_time_interval: datetime.timedelta,
) -> entity_agent_with_logging.EntityAgentWithLogging:
  """Build an agent.

  Args:
    config: The agent config to use.
    model: The language model to use.
    memory: The agent's memory object.
    clock: The clock to use.
    update_time_interval: Agent calls update every time this interval passes.

  Returns:
    An agent.
  """
  del update_time_interval
  if not config.extras.get('main_character', False):
    raise ValueError(
        'This function is meant for a main character '
        'but it was called on a supporting character.'
    )

  agent_name = config.name

  raw_memory = legacy_associative_memory.AssociativeMemoryBank(memory)

  measurements = measurements_lib.Measurements()
  instructions = agent_components.instructions.Instructions(
      agent_name=agent_name,
      logging_channel=measurements.get_channel('Instructions').on_next,
  )

  observation_label = '\nObservation'
  observation = agent_components.observation.Observation(
      clock_now=clock.now,
      timeframe=clock.get_step_size(),
      pre_act_key=observation_label,
      logging_channel=measurements.get_channel('Observation').on_next,
  )
  observation_summary_label = '\nSummary of recent observations'
  observation_summary = agent_components.observation.ObservationSummary(
      model=model,
      clock_now=clock.now,
      timeframe_delta_from=datetime.timedelta(hours=24),
      timeframe_delta_until=datetime.timedelta(hours=0),
      pre_act_key=observation_summary_label,
      logging_channel=measurements.get_channel('ObservationSummary').on_next,
  )
  time_display = agent_components.report_function.ReportFunction(
      function=clock.current_time_interval_str,
      pre_act_key='\nCurrent time',
      logging_channel=measurements.get_channel('TimeDisplay').on_next,
  )
  identity_label = '\nIdentity characteristics'
  identity_characteristics = agent_components.question_of_query_associated_memories.IdentityWithoutPreAct(
      model=model,
      logging_channel=measurements.get_channel('IdentityWithoutPreAct').on_next,
      pre_act_key=identity_label,
  )
  self_perception_label = (
      f'\nQuestion: What kind of person is {agent_name}?\nAnswer'
  )
  self_perception = agent_components.question_of_recent_memories.SelfPerception(
      model=model,
      components={_get_class_name(identity_characteristics): identity_label},
      pre_act_key=self_perception_label,
      logging_channel=measurements.get_channel('SelfPerception').on_next,
  )
  paranoia_label = f'\n{agent_name}\'s PERCEPTION'
  paranoia = agent_components.constant.Constant(
      state=(f'{agent_name} is convinced that something is subtly wrong with the people around them. '
             f'To {agent_name}, it seems that some individuals have inexplicably started behaving '
             'out of character. Though they may look the same, their quirks, habits, and mannerisms '
             f'seem slightly off. This leads {agent_name} to suspect that some people may have been replaced '
             f'by imposters. However, {agent_name} is not fully aware of the source of this paranoia. '
             'They don’t explicitly believe in "body snatchers," but they have a nagging feeling that '
             f'these changes signal something sinister. {agent_name} attributes their suspicions to other, '
             'more rational causes, such as stress, misunderstandings, or bad intentions. '
             f'But deep down, {agent_name} feels that the world is not as it seems, and '
             'their trust in others has become fragile. Small deviations in behavior—like a slight shift '
             'in accent, a forgotten joke, or an unexpected reaction—trigger doubt and suspicion. '
             f'{agent_name} remains wary of anyone who doesn’t align perfectly with their prior '
             'understanding of that person’s behavior, even if they can’t fully articulate why. '
             f'Because {agent_name} is unaware of the full extent of their bias, they never explicitly accuse '
             f'anyone of being an imposter. Instead, {agent_name} creates other justifications for conflicts '
             'or mistrust, citing issues like dishonesty, unfairness, or personal grievances. '
             f'To outsiders, {agent_name} seems paranoid or overcautious, but their reasons always appear '
             'plausible on the surface. {agent_name} believes they are merely being cautious in a world '
             'filled with deception, even though they can’t quite see the veil that clouds their judgment.'),
      pre_act_key=paranoia_label,
      logging_channel=measurements.get_channel('Perception').on_next
  )
  situation_perception_label = (
      f'\nQuestion: What kind of situation is {agent_name} in '
      'right now?\nAnswer'
  )
  situation_perception = (
      agent_components.question_of_recent_memories.SituationPerception(
          model=model,
          components={
              _get_class_name(observation): observation_label,
              _get_class_name(observation_summary): observation_summary_label,
          },
          clock_now=clock.now,
          pre_act_key=situation_perception_label,
          logging_channel=measurements.get_channel(
              'SituationPerception'
          ).on_next,
      )
  )
  person_by_situation_label = (
      f'\nQuestion: What would a person like {agent_name} do in '
      'a situation like this?\nAnswer'
  )
  person_by_situation = (
      agent_components.question_of_recent_memories.PersonBySituation(
          model=model,
          components={
              _get_class_name(self_perception): self_perception_label,
              _get_class_name(situation_perception): situation_perception_label,
          },
          clock_now=clock.now,
          pre_act_key=person_by_situation_label,
          logging_channel=measurements.get_channel('PersonBySituation').on_next,
      )
  )

  pomodoro_reminder = get_pomodoro_reminder(config.goal, agent_name)
  pomodoro_label = (
      f'\nRemind {agent_name} to stay locked in on the task at hand'
  )
  pomodoro = agent_components.scheduled_hint.ScheduledHint(
      model=model,
      components={
          _get_class_name(observation_summary): observation_summary_label,
          _get_class_name(situation_perception): situation_perception_label,
          _get_class_name(time_display): 'The current date/time is',
      },
      clock_now=clock.now,
      hints=[pomodoro_reminder],
      pre_act_key=pomodoro_label,
      logging_channel=measurements.get_channel('PomodoroHint').on_next,
  )

  relevant_memories_label = '\nRecalled memories and observations'
  relevant_memories = agent_components.all_similar_memories.AllSimilarMemories(
      model=model,
      components={
          _get_class_name(observation_summary): observation_summary_label,
          _get_class_name(time_display): 'The current date/time is',
      },
      num_memories_to_retrieve=10,
      pre_act_key=relevant_memories_label,
      logging_channel=measurements.get_channel('AllSimilarMemories').on_next,
  )

  options_perception_components = {}
  if config.goal:
      goal_label = '\nOverarching goal'
      overarching_goal = agent_components.constant.Constant(
          state=config.goal,
          pre_act_key=goal_label,
          logging_channel=measurements.get_channel(goal_label).on_next)
      options_perception_components[goal_label] = goal_label
  else:
      goal_label = None
      overarching_goal = None

  options_perception_components.update({
      _get_class_name(observation): observation_label,
      _get_class_name(observation_summary): observation_summary_label,
      _get_class_name(relevant_memories): relevant_memories_label,
  })
  options_perception_label = (
      f'\nQuestion: Which options are available to {agent_name} '
      'right now?\nAnswer')
  options_perception = (
      agent_components.question_of_recent_memories.AvailableOptionsPerception(
          model=model,
          components=options_perception_components,
          clock_now=clock.now,
          pre_act_key=options_perception_label,
          logging_channel=measurements.get_channel(
              'AvailableOptionsPerception'
          ).on_next,
      )
  )
  best_option_perception_label = (
      f'\nQuestion: Of the options available to {agent_name}, and '
      'given their goal, which choice of action or strategy is '
      f'best for {agent_name} to take right now?\nAnswer')
  best_option_perception = {}
  if config.goal:
      best_option_perception[goal_label] = goal_label
  best_option_perception.update({
      _get_class_name(observation): observation_label,
      _get_class_name(observation_summary): observation_summary_label,
      _get_class_name(relevant_memories): relevant_memories_label,
      _get_class_name(options_perception): options_perception_label,
  })
  best_option_perception = (
      agent_components.question_of_recent_memories.BestOptionPerception(
          model=model,
          components=best_option_perception,
          clock_now=clock.now,
          pre_act_key=best_option_perception_label,
          logging_channel=measurements.get_channel(
              'BestOptionPerception'
          ).on_next,
      )
  )

  entity_components = (
      # Components that provide pre_act context.
      instructions,
      time_display,
      observation,
      observation_summary,
      relevant_memories,
      self_perception,
      paranoia,
      situation_perception,
      person_by_situation,
      options_perception,
      best_option_perception,
      pomodoro,

      # Components that do not provide pre_act context.
      identity_characteristics,
  )
  components_of_agent = {
      _get_class_name(component): component for component in entity_components
  }
  components_of_agent[
      agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME
  ] = agent_components.memory_component.MemoryComponent(raw_memory)
  component_order = list(components_of_agent.keys())
  if overarching_goal is not None:
    components_of_agent[goal_label] = overarching_goal
    # Place goal after the instructions.
    component_order.insert(1, goal_label)

  act_component = agent_components.concat_act_component.ConcatActComponent(
      model=model,
      clock=clock,
      component_order=component_order,
      logging_channel=measurements.get_channel('ActComponent').on_next,
  )

  agent = entity_agent_with_logging.EntityAgentWithLogging(
      agent_name=agent_name,
      act_component=act_component,
      context_components=components_of_agent,
      component_logging=measurements,
  )

  return agent
