import sys
sys.path.append('/media/mlr_lab/6E18DC183015F19C/Ashu/Ashutosh_Dataset/VLN/Docker_Base/MapGPT')

from GPT.multi_agent_nav import Agent, PerceptionAgent, MapAgent, MemoryAgent, ActionAgent, PlanningAgent
from typing import List, Dict, Any
from openai import OpenAI
import asyncio

from collections import defaultdict
from .agent_base import BaseAgent
import pprint
import numpy as np

class CoordinatorAgent(Agent):
    def __init__(self, name: str, llm_client: OpenAI, args: Any):
        super().__init__(name, llm_client)
        self.args = args
        self.perception_agent = PerceptionAgent("Perception", llm_client)
        self.map_agent = MapAgent("Map", llm_client)
        self.planning_agent = PlanningAgent("Planning", llm_client)
        self.action_agent = ActionAgent(args)
        self.memory_agent = MemoryAgent("Memory", llm_client, args.batch_size,)

    async def process(self, obs: Dict[str, Any], t: int) -> int:
        # Process perception
        perception_result = await self.perception_agent.process(obs)
        
        # Update map
        map_update = await self.map_agent.process({
            'obs': obs,
        })
        
        # Update memory
        memory_update = await self.memory_agent.process({
            'batch_index': 0,  # Assuming single batch for simplicity
            'action': obs.get('action', ''),
            'new_node': obs['viewpoint'],
            'new_node_img': obs.get('image'),
            'connected_nodes': [c['viewpointId'] for c in obs.get('candidate', [])],
            'new_plan': ''  # Will be updated later
        })
        
        # Generate or update plan
        graph_description = await self.map_agent._generate_map_description()
        plan = await self.planning_agent.process({
            'instruction': obs['instruction'],
            'graph_description': graph_description,
            'current_position': obs['viewpoint'],
            'perception_data': perception_result
        })
        
        # Update memory with new plan
        await self.memory_agent.update_planning(0, plan['plan_summary'])
        
        # Decide on action
        action_options = self._get_action_options(obs['candidate'])
        action_index = await self.action_agent.process(
            plan['plan'][0],  # Use the first step of the plan
            action_options,
            obs,
            t
        )
        
        # Log the decision process
        await self.log(f"Step {t}: Perception processed, map updated, plan generated, action decided: {action_options[action_index]}")
        
        return action_index

    def _get_action_options(self, candidates):
        action_options = []
        for candidate in candidates:
            rel_heading = candidate.get('rel_heading', 0)
            rel_elevation = candidate.get('rel_elevation', 0)
            action = self.action_agent.get_action_concept(rel_heading, rel_elevation)
            action_options.append(action)
        return action_options
    
    async def initialize(self) -> None:
        await super().initialize()
        await asyncio.gather(
            self.perception_agent.initialize(),
            self.map_agent.initialize(),
            self.planning_agent.initialize(),
            self.memory_agent.initialize()
        )

    async def shutdown(self) -> None:
        await asyncio.gather(
            self.perception_agent.shutdown(),
            self.map_agent.shutdown(),
            self.planning_agent.shutdown(),
            self.memory_agent.shutdown()
        )
        await super().shutdown()

class MultiAgentNavigationSystem:
    def __init__(self, args: Any):
        self.args = args
        self.llm_client = OpenAI(
            base_url='http://localhost:11434/v1',
            api_key='ollama',
        )
        self.coordinator = CoordinatorAgent("Coordinator", self.llm_client, args)

    async def navigate(self, obs: Dict[str, Any], t: int) -> int:
        return await self.coordinator.process(obs, t)

    async def initialize(self) -> None:
        await self.coordinator.initialize()

    async def shutdown(self) -> None:
        await self.coordinator.shutdown()

class MultiAgentGPTNavAgent(BaseAgent):
    env_actions = {
        'left': (0, -1, 0),  # left
        'right': (0, 1, 0),  # right
        'up': (0, 0, 1),  # up
        'down': (0, 0, -1),  # down
        'forward': (1, 0, 0),  # forward
        '<end>': (0, 0, 0),  # <end>
        '<start>': (0, 0, 0),  # <start>
        '<ignore>': (0, 0, 0)  # <ignore>
    }
    for k, v in env_actions.items():
        env_actions[k] = [[vx] for vx in v]

    def __init__(self, args, env, rank=0):
        super().__init__(env)
        self.args = args
        self.multi_agent_system = MultiAgentNavigationSystem(args)
        self.logs = defaultdict(list)

    async def rollout(self, train_ml=None, train_rl=False, reset=True):
        if reset:
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()

        print("----------------------------------------OBS----------------------------------------")
        pprint.pformat(obs)
        batch_size = len(obs)

        traj = [{
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'details': {},
            'a_t': {},
        } for ob in obs]

        if traj[0]['instr_id'] in self.results:
            return [None]

        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size)

        await self.multi_agent_system.initialize()

        for t in range(self.args.max_action_len):
            if t == self.args.max_action_len:
                break

            a_t = await self.multi_agent_system.navigate(obs[0], t)

            for i in range(batch_size):
                traj[i]['a_t'][t] = a_t

            a_t_stop = [a_t == 0]

            cpu_a_t = []
            for i in range(batch_size):
                if a_t_stop[i] or ended[i]:
                    cpu_a_t.append(-1)
                    just_ended[i] = True
                else:
                    cpu_a_t.append(a_t)

            self.make_equiv_action(cpu_a_t, obs, traj)
            obs = self.env._get_obs()

            if a_t == 0:
                break

        # await self.multi_agent_system.shutdown()
        return traj

    def make_equiv_action(self, a_t, obs, traj=None):
        def take_action(i, name):
            if type(name) is int:
                self.env.env.sims[i].makeAction([name], [0], [0])
            else:
                self.env.env.sims[i].makeAction(*self.env_actions[name])

        for i, ob in enumerate(obs):
            action = a_t[i]
            if action != -1:
                select_candidate = ob['candidate'][action]
                src_point = ob['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point) // 12
                trg_level = (trg_point) // 12
                while src_level < trg_level:
                    take_action(i, 'up')
                    src_level += 1
                while src_level > trg_level:
                    take_action(i, 'down')
                    src_level -= 1
                while self.env.env.sims[i].getState()[0].viewIndex != trg_point:
                    take_action(i, 'right')
                assert select_candidate['viewpointId'] == \
                       self.env.env.sims[i].getState()[0].navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, select_candidate['idx'])

                state = self.env.env.sims[i].getState()[0]
                if traj is not None:
                    traj[i]['path'].append([state.location.viewpointId])