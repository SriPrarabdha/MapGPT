import re
from typing import List, Dict, Any
import asyncio

import math
import random
from abc import ABC, abstractmethod

class Agent(ABC):
     def __init__(self, name: str):
        self.name = name


class PerceptionAgent(Agent):
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def process(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the observation data, including images, and extract relevant information.
        
        :param obs: A dictionary containing observation data, including images
        :return: A dictionary with processed perception data
        """
        perception_results = {}

        # Process the current viewpoint image
        current_image = obs.get('image')
        if current_image:
            perception_results['current_view'] = await self._process_image(current_image)

        # Process candidate images
        candidate_perceptions = []
        for candidate in obs.get('candidate', []):
            candidate_image = candidate.get('image')
            if candidate_image:
                candidate_perception = await self._process_image(candidate_image)
                candidate_perception['viewpointId'] = candidate['viewpointId']
                candidate_perception['pointId'] = candidate['pointId']
                candidate_perceptions.append(candidate_perception)

        perception_results['candidates'] = candidate_perceptions

        return perception_results

    async def _process_image(self, image: Any) -> Dict[str, Any]:
        """
        Process a single image using the LLM with vision support.
        
        :param image: The image data
        :return: A dictionary containing the processed image information
        """
        prompt = """
        Analyze this image and provide a detailed description. Focus on the following aspects:
        1. Overall scene description (e.g., indoor/outdoor, room type)
        2. Key objects and their positions
        3. Potential landmarks or notable features
        4. Any text visible in the image
        5. Estimated distances to major objects
        6. Potential navigation paths or obstacles
        """

        response = await self.llm_client.process_image(image, prompt)
        
        # Parse the LLM response to extract structured information
        # This is a simplified parsing; you might want to use more sophisticated NLP techniques
        parsed_info = {
            'scene_description': self._extract_info(response, 'Overall scene description:'),
            'key_objects': self._extract_list(response, 'Key objects:'),
            'landmarks': self._extract_list(response, 'Potential landmarks:'),
            'visible_text': self._extract_info(response, 'Visible text:'),
            'distances': self._extract_info(response, 'Estimated distances:'),
            'navigation_info': self._extract_info(response, 'Potential navigation paths or obstacles:')
        }

        return parsed_info

    def _extract_info(self, text: str, key: str) -> str:
        """Extract information for a given key from the text."""
        start = text.find(key)
        if start != -1:
            end = text.find('\n', start)
            return text[start + len(key):end].strip() if end != -1 else text[start + len(key):].strip()
        return ""

    def _extract_list(self, text: str, key: str) -> List[str]:
        """Extract a list of items for a given key from the text."""
        info = self._extract_info(text, key)
        return [item.strip() for item in info.split(',') if item.strip()]

class MapAgent(Agent):
    def __init__(self, batch_size):
        self.nodes_list = [[] for _ in range(batch_size)]
        self.node_imgs = [[] for _ in range(batch_size)]
        self.graph = [{} for _ in range(batch_size)]
        self.trajectory = [[] for _ in range(batch_size)]

    async def process(self, obs, cand_inputs):
        await asyncio.gather(*[self.update_map(i, ob, cand_inputs) for i, ob in enumerate(obs)])
        return self.get_map_data()

    async def update_map(self, i, ob, cand_inputs):
        # Update nodes list
        if ob['viewpoint'] not in self.nodes_list[i]:
            self.nodes_list[i].append(ob['viewpoint'])
            self.node_imgs[i].append(None)

        # Update trajectory
        self.trajectory[i].append(ob['viewpoint'])

        # Update graph
        if ob['viewpoint'] not in self.graph[i]:
            self.graph[i][ob['viewpoint']] = []

        # Update candidates
        for j, cc in enumerate(ob['candidate']):
            if cc['viewpointId'] not in self.nodes_list[i]:
                self.nodes_list[i].append(cc['viewpointId'])
                self.node_imgs[i].append(cc['image'])
            else:
                node_index = self.nodes_list[i].index(cc['viewpointId'])
                self.node_imgs[i][node_index] = cc['image']

            if cc['viewpointId'] not in self.graph[i][ob['viewpoint']]:
                self.graph[i][ob['viewpoint']].append(cc['viewpointId'])

    def get_map_data(self):
        return {
            'nodes_list': self.nodes_list,
            'node_imgs': self.node_imgs,
            'graph': self.graph,
            'trajectory': self.trajectory
        }

    def make_map_prompt(self, i):
        trajectory = self.trajectory[i]
        nodes_list = self.nodes_list[i]
        graph = self.graph[i]

        no_dup_nodes = []
        trajectory_text = 'Place'
        graph_text = ''

        candidate_nodes = graph[trajectory[-1]]

        # Trajectory and map connectivity
        for node in trajectory:
            node_index = nodes_list.index(node)
            trajectory_text += f" {node_index}"

            if node not in no_dup_nodes:
                no_dup_nodes.append(node)

                adj_text = ''
                adjacent_nodes = graph[node]
                for adj_node in adjacent_nodes:
                    adj_index = nodes_list.index(adj_node)
                    adj_text += f" {adj_index},"

                graph_text += f"\nPlace {node_index} is connected with Places{adj_text}"[:-1]

        # Ghost nodes info
        graph_supp_text = ''
        supp_exist = None
        for node_index, node in enumerate(nodes_list):
            if node in trajectory or node in candidate_nodes:
                continue
            supp_exist = True
            graph_supp_text += f"\nPlace {node_index}, which is corresponding to Image {node_index}"

        if supp_exist is None:
            graph_supp_text = "Nothing yet."

        return trajectory_text, graph_text, graph_supp_text



class MemoryAgent(Agent):
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.history: List[str] = ['' for _ in range(self.batch_size)]
        self.nodes_list: List[List[str]] = [[] for _ in range(self.batch_size)]
        self.node_imgs: List[List[Any]] = [[] for _ in range(self.batch_size)]
        self.graph: List[Dict[str, List[str]]] = [{} for _ in range(self.batch_size)]
        self.trajectory: List[List[str]] = [[] for _ in range(self.batch_size)]
        self.planning: List[List[str]] = [["Navigation has just started, with no planning yet."] for _ in range(self.batch_size)]

    async def process(self, new_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process new information and update the agent's memory.
        
        :param new_info: Dictionary containing new information to be processed
        :return: Dictionary containing updated memory state
        """
        batch_index = new_info.get('batch_index', 0)
        obs = new_info.get('obs', {})
        action = new_info.get('action', '')
        t = new_info.get('t', 0)

        # Update nodes list and trajectory
        if obs['viewpoint'] not in self.nodes_list[batch_index]:
            self.nodes_list[batch_index].append(obs['viewpoint'])
            self.node_imgs[batch_index].append(None)
        self.trajectory[batch_index].append(obs['viewpoint'])

        # Update graph
        if obs['viewpoint'] not in self.graph[batch_index]:
            self.graph[batch_index][obs['viewpoint']] = [c['viewpointId'] for c in obs['candidate']]

        # Update history
        if t == 0:
            self.history[batch_index] = f"step {t}: {action}"
        else:
            self.history[batch_index] += f", step {t}: {action}"

        # Update planning
        if 'planning' in new_info:
            planning = new_info['planning'].replace('new', 'previous').replace('New', 'Previous')
            self.planning[batch_index].append(planning)

        return {
            'history': self.history[batch_index],
            'nodes_list': self.nodes_list[batch_index],
            'graph': self.graph[batch_index],
            'trajectory': self.trajectory[batch_index],
            'planning': self.planning[batch_index][-1]
        }

    async def get_memory_state(self, batch_index: int) -> Dict[str, Any]:
        """
        Retrieve the current memory state for a given batch index.
        
        :param batch_index: Index of the batch to retrieve memory for
        :return: Dictionary containing the current memory state
        """
        return {
            'history': self.history[batch_index],
            'nodes_list': self.nodes_list[batch_index],
            'graph': self.graph[batch_index],
            'trajectory': self.trajectory[batch_index],
            'planning': self.planning[batch_index][-1]
        }

    async def clear_memory(self, batch_index: int) -> None:
        """
        Clear the memory for a given batch index.
        
        :param batch_index: Index of the batch to clear memory for
        """
        self.history[batch_index] = ''
        self.nodes_list[batch_index] = []
        self.node_imgs[batch_index] = []
        self.graph[batch_index] = {}
        self.trajectory[batch_index] = []
        self.planning[batch_index] = ["Navigation has just started, with no planning yet."]

    async def make_map_prompt(self, batch_index: int) -> Tuple[str, str, str]:
        """
        Generate map-related prompts for the given batch index.
        
        :param batch_index: Index of the batch to generate prompts for
        :return: Tuple containing trajectory_text, graph_text, and graph_supp_text
        """
        trajectory = self.trajectory[batch_index]
        nodes_list = self.nodes_list[batch_index]
        graph = self.graph[batch_index]

        no_dup_nodes = []
        trajectory_text = 'Place'
        graph_text = ''

        candidate_nodes = graph[trajectory[-1]]

        # Trajectory and map connectivity
        for node in trajectory:
            node_index = nodes_list.index(node)
            trajectory_text += f" {node_index}"

            if node not in no_dup_nodes:
                no_dup_nodes.append(node)

                adj_text = ''
                adjacent_nodes = graph[node]
                for adj_node in adjacent_nodes:
                    adj_index = nodes_list.index(adj_node)
                    adj_text += f" {adj_index},"

                graph_text += f"\nPlace {node_index} is connected with Places{adj_text}"[:-1]

        # Ghost nodes info
        graph_supp_text = ''
        supp_exist = None
        for node_index, node in enumerate(nodes_list):
            if node in trajectory or node in candidate_nodes:
                continue
            supp_exist = True
            graph_supp_text += f"\nPlace {node_index}, which is corresponding to Image {node_index}"

        if supp_exist is None:
            graph_supp_text = "Nothing yet."

        return trajectory_text, graph_text, graph_supp_text
    
class ActionAgent(Agent):
    def __init__(self, args):
        self.args = args

    async def process(self, plan: str, action_options: List[str], obs: Dict[str, Any], t: int) -> int:
        """
        Decide on the next action based on the current plan and available options.
        
        :param plan: The current navigation plan
        :param action_options: List of available actions
        :param obs: Current observation
        :param t: Current time step
        :return: Index of the chosen action
        """
        # Parse the plan to extract the next intended action
        next_action = self._parse_plan(plan)
        
        # Match the intended action with available options
        action_index = self._match_action(next_action, action_options)
        
        # If no match found, use a fallback strategy
        if action_index is None:
            action_index = self._fallback_strategy(obs, action_options, t)
        
        return action_index

    def _parse_plan(self, plan: str) -> str:
        """
        Extract the next intended action from the plan.
        """
        # Simple parsing strategy: take the first sentence as the next action
        sentences = plan.split('.')
        if sentences:
            return sentences[0].strip().lower()
        return ""

    def _match_action(self, intended_action: str, action_options: List[str]) -> int:
        """
        Match the intended action with available options.
        """
        for i, option in enumerate(action_options):
            if self._action_matches(intended_action, option.lower()):
                return i
        return None

    def _action_matches(self, intended_action: str, option: str) -> bool:
        """
        Check if the intended action matches an option.
        """
        # Define key phrases for each action type
        action_phrases = {
            "go forward": ["go forward", "move forward", "continue straight"],
            "turn left": ["turn left", "go left"],
            "turn right": ["turn right", "go right"],
            "turn around": ["turn around", "go back"],
            "go up": ["go up", "move up", "ascend"],
            "go down": ["go down", "move down", "descend"],
            "stop": ["stop", "halt", "end navigation"]
        }

        # Check if any key phrase for the option is in the intended action
        for action, phrases in action_phrases.items():
            if action in option:
                return any(phrase in intended_action for phrase in phrases)

        return False

    def _fallback_strategy(self, obs: Dict[str, Any], action_options: List[str], t: int) -> int:
        """
        Implement a fallback strategy when the plan doesn't match available actions.
        """
        # Example fallback: choose the action that brings us closest to the goal
        goal_position = self._extract_goal_position(obs['instruction'])
        current_position = obs['viewpoint']
        
        best_action = 0
        min_distance = float('inf')
        
        for i, option in enumerate(action_options):
            next_position = self._simulate_action(current_position, option)
            distance = self._calculate_distance(next_position, goal_position)
            
            if distance < min_distance:
                min_distance = distance
                best_action = i
        
        # Avoid stopping in the first few steps
        if bool(self.args.stop_after) and t < self.args.stop_after:
            best_action = max(1, best_action)  # Ensure we don't choose 'stop' (assumed to be at index 0)
        
        return best_action

    def _extract_goal_position(self, instruction: str) -> Dict[str, float]:
        """
        Extract the goal position from the instruction.
        This is a placeholder and should be implemented based on your specific instruction format.
        """
        # Placeholder implementation
        return {'x': 0, 'y': 0, 'z': 0}

    def _simulate_action(self, current_position: Dict[str, float], action: str) -> Dict[str, float]:
        """
        Simulate the result of taking an action from the current position.
        This is a placeholder and should be implemented based on your specific environment.
        """
        # Placeholder implementation
        return current_position

    def _calculate_distance(self, pos1: Dict[str, float], pos2: Dict[str, float]) -> float:
        """
        Calculate the distance between two positions.
        """
        return math.sqrt(sum((pos1[k] - pos2[k])**2 for k in pos1.keys()))

    def get_action_concept(self, rel_heading, rel_elevation):
        """
        Get the action concept based on relative heading and elevation.
        """
        if rel_elevation > 0:
            action_text = 'go up'
        elif rel_elevation < 0:
            action_text = 'go down'
        else:
            if rel_heading < 0:
                if rel_heading >= -math.pi / 2:
                    action_text = 'turn left'
                elif rel_heading < -math.pi / 2 and rel_heading > -math.pi * 3 / 2:
                    action_text = 'turn around'
                else:
                    action_text = 'turn right'
            elif rel_heading > 0:
                if rel_heading <= math.pi / 2:
                    action_text = 'turn right'
                elif rel_heading > math.pi / 2 and rel_heading < math.pi * 3 / 2:
                    action_text = 'turn around'
                else:
                    action_text = 'turn left'
            elif rel_heading == 0:
                action_text = 'go forward'

        return action_text
    
class PlanningAgent(Agent):
    def __init__(self):
        self.current_plan: List[str] = []
        self.instruction: str = ""
        self.trajectory: List[str] = []
        self.graph: Dict[str, List[str]] = {}
        self.history: str = ""

    async def process(self, instruction: str, map_data: Dict[str, Any], current_position: str) -> List[str]:
        self.instruction = instruction
        self.trajectory = map_data['trajectory']
        self.graph = map_data['graph']
        self.history = map_data['history']
        
        return await self.generate_plan(current_position)

    async def generate_plan(self, current_position: str) -> List[str]:
        # Analyze the instruction and current state
        remaining_instructions = self.extract_remaining_instructions()
        
        # Generate a high-level plan
        high_level_plan = self.create_high_level_plan(remaining_instructions)
        
        # Translate high-level plan to specific actions
        detailed_plan = self.translate_to_detailed_plan(high_level_plan, current_position)
        
        self.current_plan = detailed_plan
        return self.current_plan

    def extract_remaining_instructions(self) -> List[str]:
        executed_steps = self.parse_history()
        instruction_steps = self.instruction.split('.')
        
        remaining_steps = []
        for step in instruction_steps:
            if not any(executed_step in step for executed_step in executed_steps):
                remaining_steps.append(step.strip())
        
        return remaining_steps

    def parse_history(self) -> List[str]:
        executed_steps = []
        step_pattern = re.compile(r'step (\d+): (.+?)(?=, step \d+:|$)')
        matches = step_pattern.findall(self.history)
        
        for _, action in matches:
            executed_steps.append(action)
        
        return executed_steps

    def create_high_level_plan(self, remaining_instructions: List[str]) -> List[str]:
        high_level_plan = []
        for instruction in remaining_instructions:
            if "go to" in instruction.lower():
                high_level_plan.append(f"Navigate: {instruction}")
            elif "turn" in instruction.lower():
                high_level_plan.append(f"Rotate: {instruction}")
            elif "stop" in instruction.lower() or "you have reached" in instruction.lower():
                high_level_plan.append("Stop: End navigation")
            else:
                high_level_plan.append(f"Explore: {instruction}")
        return high_level_plan

    def translate_to_detailed_plan(self, high_level_plan: List[str], current_position: str) -> List[str]:
        detailed_plan = []
        for step in high_level_plan:
            if step.startswith("Navigate:"):
                target = self.extract_target(step)
                path = self.find_path(current_position, target)
                detailed_plan.extend(path)
                current_position = target
            elif step.startswith("Rotate:"):
                direction = "left" if "left" in step.lower() else "right"
                detailed_plan.append(f"Turn {direction}")
            elif step.startswith("Stop:"):
                detailed_plan.append("Stop navigation")
            elif step.startswith("Explore:"):
                detailed_plan.append(f"Explore surroundings: {step.split(': ')[1]}")
        return detailed_plan

    def extract_target(self, navigation_step: str) -> str:
        # Simple extraction, can be improved with NLP techniques
        return navigation_step.split("go to ")[-1].strip()

    def find_path(self, start: str, goal: str) -> List[str]:
        # Implement a pathfinding algorithm (e.g., A* or Dijkstra's)
        # This is a simplified version using BFS
        queue = [(start, [start])]
        visited = set()

        while queue:
            (node, path) = queue.pop(0)
            if node not in visited:
                if node == goal:
                    return self.path_to_actions(path)
                visited.add(node)
                for neighbor in self.graph.get(node, []):
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))
        
        return ["Unable to find path to " + goal]

    def path_to_actions(self, path: List[str]) -> List[str]:
        actions = []
        for i in range(len(path) - 1):
            current = path[i]
            next_node = path[i + 1]
            action = f"Move from Place {self.trajectory.index(current)} to Place {self.trajectory.index(next_node)}"
            actions.append(action)
        return actions
    

class CoordinatorAgent(Agent):
    def __init__(self):
        self.perception_agent = PerceptionAgent()
        self.map_agent = MapAgent()
        self.planning_agent = PlanningAgent()
        self.action_agent = ActionAgent()
        self.memory_agent = MemoryAgent()

    async def process(self, obs, cand_inputs, t):
        # Coordinate the flow of information between agents
        perception_result = await self.perception_agent.process(obs)
        map_update = await self.map_agent.process(self.trajectory, self.graph)
        memory_update = await self.memory_agent.process(perception_result)
        plan = await self.planning_agent.process(obs['instruction'], map_update, self.current_position)
        action = await self.action_agent.process(plan, cand_inputs['action_prompts'])
        return action

class MultiAgentNavigationSystem:
    def __init__(self, args):
        self.args = args
        self.coordinator = CoordinatorAgent()

    async def navigate(self, obs, cand_inputs, t):
        return await self.coordinator.process(obs, cand_inputs, t)

# Usage
async def main():
    batch_size = 2
    map_agent = MapAgent(batch_size)
    
    # Sample data (you would replace this with actual data from your navigation system)
    obs = [
        {
            'viewpoint': 'A',
            'candidate': [
                {'viewpointId': 'B', 'image': 'image_B'},
                {'viewpointId': 'C', 'image': 'image_C'}
            ]
        },
        {
            'viewpoint': 'X',
            'candidate': [
                {'viewpointId': 'Y', 'image': 'image_Y'},
                {'viewpointId': 'Z', 'image': 'image_Z'}
            ]
        }
    ]
    cand_inputs = {}  # Add relevant candidate inputs if needed

    map_data = await map_agent.process(obs, cand_inputs)
    print("Updated Map Data:", map_data)

    # Generate map prompts for each batch
    for i in range(batch_size):
        trajectory_text, graph_text, graph_supp_text = map_agent.make_map_prompt(i)
        print(f"\nBatch {i}:")
        print("Trajectory:", trajectory_text)
        print("Graph:", graph_text)
        print("Supplementary Info:", graph_supp_text)

# Implement MCTS 


class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

class MCTS:
    def __init__(self, exploration_weight=1.4):
        self.exploration_weight = exploration_weight

    def choose_action(self, root_state, num_simulations):
        root = MCTSNode(root_state)
        for _ in range(num_simulations):
            node = self.select(root)
            reward = self.simulate(node.state)
            self.backpropagate(node, reward)
        return self.best_child(root).state.last_action

    def select(self, node):
        while node.children:
            if len(node.children) < len(node.state.get_legal_actions()):
                return self.expand(node)
            else:
                node = self.best_uct(node)
        return node

    def expand(self, node):
        tried_actions = [c.state.last_action for c in node.children]
        untried_actions = [a for a in node.state.get_legal_actions() if a not in tried_actions]
        action = random.choice(untried_actions)
        child_state = node.state.take_action(action)
        child = MCTSNode(child_state, parent=node)
        node.children.append(child)
        return child

    def simulate(self, state):
        while not state.is_terminal():
            action = random.choice(state.get_legal_actions())
            state = state.take_action(action)
        return state.get_reward()

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def best_uct(self, node):
        return max(node.children, key=lambda c: self.uct_value(c))

    def uct_value(self, node):
        if node.visits == 0:
            return float('inf')
        return node.value / node.visits + self.exploration_weight * math.sqrt(
            math.log(node.parent.visits) / node.visits)

    def best_child(self, node):
        return max(node.children, key=lambda c: c.visits)

class NavigationState:
    def __init__(self, current_position, goal_position, environment):
        self.current_position = current_position
        self.goal_position = goal_position
        self.environment = environment
        self.last_action = None

    def get_legal_actions(self):
        # Return list of legal actions based on current position and environment
        pass

    def take_action(self, action):
        # Return a new NavigationState after taking the action
        pass

    def is_terminal(self):
        # Check if goal is reached or max steps are taken
        pass

    def get_reward(self):
        # Return reward based on proximity to goal, obstacles avoided, etc.
        pass

class MCTSNavigator:
    def __init__(self, num_simulations=1000):
        self.mcts = MCTS()
        self.num_simulations = num_simulations

    def choose_action(self, current_state):
        return self.mcts.choose_action(current_state, self.num_simulations)
if __name__ == "__main__":
    asyncio.run(main())