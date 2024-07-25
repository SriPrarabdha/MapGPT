import re
from typing import List, Dict, Any
import asyncio
from openai import OpenAI

import math
import random
from abc import ABC, abstractmethod
import base64

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

class Agent(ABC):
    """
    Abstract base class for all agents in the multi-agent navigation system.
    """

    def __init__(self, name: str, llm_client: OpenAI):
        """
        Initialize the agent with a name and LLM client.

        :param name: A string identifier for the agent
        :param llm_client: An instance of LLMClient for interacting with the LLM
        """
        self.name = name
        self.llm_client = OpenAI(
        base_url = 'http://localhost:11434/v1',
        api_key='ollama', # required, but unused
    )

    @abstractmethod
    async def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Process the input data and return a result.

        This method should be implemented by all subclasses.

        :param input_data: The input data to be processed. The type can vary depending on the specific agent.
        :return: A dictionary containing the processing results.
        """
        pass

    async def initialize(self) -> None:
        """
        Initialize the agent's state or perform any necessary setup.

        This method can be overridden by subclasses if needed.
        """
        await self.log("Initializing agent")

    async def shutdown(self) -> None:
        """
        Perform any necessary cleanup or resource release.

        This method can be overridden by subclasses if needed.
        """
        await self.log("Shutting down agent")

    def __str__(self) -> str:
        """
        Return a string representation of the agent.

        :return: A string describing the agent.
        """
        return f"{self.__class__.__name__}(name={self.name})"

    @property
    def agent_type(self) -> str:
        """
        Return the type of the agent.

        :return: A string representing the agent's type.
        """
        return self.__class__.__name__

    async def log(self, message: str) -> None:
        """
        Log a message from the agent.

        This method can be used for debugging or monitoring agent activities.

        :param message: The message to be logged.
        """
        print(f"[{self.name}] {message}")

    async def generate_with_llm(self, system, user_content, **kwargs) -> str:
        """
        Generate a response using the LLM.

        :param prompt: The prompt to send to the LLM
        :param kwargs: Additional keyword arguments for the LLM client
        :return: The generated response
        """
        messages = [
        {"role": "system",
         "content": system
         },
        {"role": "user",
         "content": user_content
         }
    ]
        response = self.llm_client.chat.completions.create(model="llava:7b",temperature=0.5, max_tokens=1000, messages=messages, **kwargs)
        print("-----------------------------open ai working -------------------------------")
        print(response.choices[0].message.content)
        return response.choices[0].message.content


class PerceptionAgent(Agent):
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
        with open(image, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        user_content=[]
        system = """You are an embodied robot that navigates in the real world."""
        image_message = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}",
                        "detail": "low"
                    }
                }
        
        user_content.append(image_message)
        user_content.append(
        {
            "type": "text",
            "text": prompt
        }
    )
        
        response = await self.generate_with_llm(system=system, user_content=user_content)
        
        # Parse the LLM response to extract structured information
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
    def __init__(self, name: str, llm_client: OpenAI):
        super().__init__(name, llm_client)
        self.nodes_list = []  # List of visited viewpoint IDs
        self.node_imgs = []   # List of images corresponding to nodes
        self.graph = {}       # Dictionary to store linguistic description of connections
        self.trajectory = []  # List of visited viewpoint IDs in order

    async def initialize(self) -> None:
        await super().initialize()
        await self.log("Initializing MapAgent")

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input data to update the map.
        :param input_data: A dictionary containing:
        - obs: The current observation data
        :return: A dictionary containing the updated map information
        """
        obs = input_data['obs']
        current_viewpoint = obs['viewpoint']

        # Update nodes list and trajectory
        if current_viewpoint not in self.nodes_list:
            self.nodes_list.append(current_viewpoint)
            self.node_imgs.append(None)  # Placeholder for image
        self.trajectory.append(current_viewpoint)

        # Update graph with connections
        self.graph[current_viewpoint] = []
        for candidate in obs['candidate']:
            cand_viewpoint = candidate['viewpointId']
            self.graph[current_viewpoint].append(cand_viewpoint)
            if cand_viewpoint not in self.nodes_list:
                self.nodes_list.append(cand_viewpoint)
                self.node_imgs.append(candidate['image'])
            else:
                node_index = self.nodes_list.index(cand_viewpoint)
                self.node_imgs[node_index] = candidate['image']

        # Generate map description
        map_description = await self._generate_map_description()

        return {
            'nodes_list': self.nodes_list,
            'node_imgs': self.node_imgs,
            'graph': self.graph,
            'trajectory': self.trajectory,
            'map_description': map_description
        }

    async def _generate_map_description(self) -> str:
        """
        Generate a textual description of the current map state using the LLM.

        :return: A string describing the current map state
        """
        trajectory_text = self._format_trajectory()
        graph_text = self._format_graph()

        prompt = f"""
        Given the following map information, provide a concise description of the environment:

        Trajectory: {trajectory_text}

        Connections:
        {graph_text}

        Describe the overall structure of the environment, highlighting the path taken and potential navigation options.
        """
        system = """You are an embodied robot that navigates in the real world."""
        user_content = []
        user_content.append(
        {
            "type": "text",
            "text": prompt
        })

        return await self.generate_with_llm(system=system, user_content=user_content)

    def _format_trajectory(self) -> str:
        """Format the trajectory for the LLM prompt."""
        return "Place " + " ".join([str(self.nodes_list.index(node)) for node in self.trajectory])

    def _format_graph(self) -> str:
        """Format the graph connections for the LLM prompt."""
        graph_text = ""
        for node, connections in self.graph.items():
            node_index = self.nodes_list.index(node)
            connection_indices = [str(self.nodes_list.index(conn)) for conn in connections]
            graph_text += f"Place {node_index} is connected with Places {', '.join(connection_indices)}\n"
        return graph_text

    async def make_map_prompt(self) -> tuple:
        """
        Create prompts for trajectory, graph, and supplementary information.

        :return: A tuple containing trajectory_text, graph_text, and graph_supp_text
        """
        trajectory_text = self._format_trajectory()
        graph_text = self._format_graph()

        # Generate supplementary info
        graph_supp_text = ""
        supp_exist = None
        for node_index, node in enumerate(self.nodes_list):
            if node in self.trajectory or node in self.graph.get(self.trajectory[-1], []):
                continue
            supp_exist = True
            graph_supp_text += f"\nPlace {node_index}, which is corresponding to Image {node_index}"

        if supp_exist is None:
            graph_supp_text = "Nothing yet."

        return trajectory_text, graph_text, graph_supp_text

    async def shutdown(self) -> None:
        await self.log("Shutting down MapAgent")
        await super().shutdown()

## code to debug
class MemoryAgent(Agent):
    def __init__(self, name: str, llm_client: OpenAI, batch_size: int, short_term_capacity: int = 10):
        super().__init__(name, llm_client)
        self.batch_size = batch_size
        self.short_term_capacity = short_term_capacity
        
        # Short-term memory
        self.short_term_memory: List[List[Dict[str, Any]]] = [[] for _ in range(self.batch_size)]
        
        # Long-term memory
        self.long_term_memory: List[List[str]] = [[] for _ in range(self.batch_size)]
        
        # Other memory structures
        self.history: List[str] = ['' for _ in range(self.batch_size)]
        self.nodes_list: List[List[str]] = [[] for _ in range(self.batch_size)]
        self.node_imgs: List[List[Any]] = [[] for _ in range(self.batch_size)]
        self.graph: List[Dict[str, List[str]]] = [{} for _ in range(self.batch_size)]
        self.trajectory: List[List[str]] = [[] for _ in range(self.batch_size)]
        self.planning: List[List[str]] = [["Navigation has just started, with no planning yet."] for _ in range(self.batch_size)]

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        batch_index = input_data.get('batch_index', 0)
        action = input_data.get('action', '')
        new_node = input_data.get('new_node', '')
        new_node_img = input_data.get('new_node_img', None)
        connected_nodes = input_data.get('connected_nodes', [])
        new_plan = input_data.get('new_plan', '')

        # Update short-term memory
        await self.update_short_term_memory(batch_index, input_data)

        # Update other memory structures
        await self.update_history(batch_index, action)
        await self.update_graph(batch_index, new_node, connected_nodes)
        await self.update_trajectory(batch_index, new_node)
        await self.update_planning(batch_index, new_plan)
        await self.update_node_images(batch_index, new_node, new_node_img)

        # Update long-term memory
        await self.update_long_term_memory(batch_index)

        return self.get_memory_state(batch_index)

    async def update_short_term_memory(self, batch_index: int, input_data: Dict[str, Any]) -> None:
        self.short_term_memory[batch_index].append(input_data)
        if len(self.short_term_memory[batch_index]) > self.short_term_capacity:
            self.short_term_memory[batch_index].pop(0)
        await self.log(f"Updated short-term memory for batch {batch_index}")

    async def update_long_term_memory(self, batch_index: int) -> None:
        # Summarize short-term memory and add to long-term memory
        summary = await self.summarize_short_term_memory(batch_index)
        self.long_term_memory[batch_index].append(summary)
        await self.log(f"Updated long-term memory for batch {batch_index}")

    async def summarize_short_term_memory(self, batch_index: int) -> str:
        short_term_data = self.short_term_memory[batch_index]
        prompt = f"Summarize the following sequence of events:\n{short_term_data}\n"
        prompt += "Provide a concise summary that captures the key information and patterns."
        user_content = []
        user_content.append(
        {
            "type": "text",
            "text": prompt
        })
        system = """You are an embodied robot that navigates in the real world."""
        summary = await self.generate_with_llm(system=system, user_content=user_content)
        return summary

    async def update_history(self, batch_index: int, action: str) -> None:
        if self.history[batch_index]:
            self.history[batch_index] += f", {action}"
        else:
            self.history[batch_index] = action
        await self.log(f"Updated history for batch {batch_index}: {self.history[batch_index]}")

    async def update_graph(self, batch_index: int, new_node: str, connected_nodes: List[str]) -> None:
        if new_node not in self.nodes_list[batch_index]:
            self.nodes_list[batch_index].append(new_node)

        if new_node not in self.graph[batch_index]:
            self.graph[batch_index][new_node] = []

        for connected_node in connected_nodes:
            if connected_node not in self.graph[batch_index][new_node]:
                self.graph[batch_index][new_node].append(connected_node)
            
            if connected_node not in self.graph[batch_index]:
                self.graph[batch_index][connected_node] = []
            if new_node not in self.graph[batch_index][connected_node]:
                self.graph[batch_index][connected_node].append(new_node)

        await self.log(f"Updated graph for batch {batch_index}: {self.graph[batch_index]}")

    async def update_trajectory(self, batch_index: int, new_node: str) -> None:
        self.trajectory[batch_index].append(new_node)
        await self.log(f"Updated trajectory for batch {batch_index}: {self.trajectory[batch_index]}")

    async def update_planning(self, batch_index: int, new_plan: str) -> None:
        self.planning[batch_index].append(new_plan)
        await self.log(f"Updated planning for batch {batch_index}: {new_plan}")

    async def update_node_images(self, batch_index: int, new_node: str, new_node_img: Any) -> None:
        if new_node not in self.nodes_list[batch_index]:
            self.nodes_list[batch_index].append(new_node)
            self.node_imgs[batch_index].append(new_node_img)
        else:
            index = self.nodes_list[batch_index].index(new_node)
            # Ensure node_imgs list is long enough
            while len(self.node_imgs[batch_index]) <= index:
                self.node_imgs[batch_index].append(None)
            self.node_imgs[batch_index][index] = new_node_img
        await self.log(f"Updated node image for batch {batch_index}, node {new_node}")

    def get_memory_state(self, batch_index: int) -> Dict[str, Any]:
        return {
            'short_term_memory': self.short_term_memory[batch_index],
            'long_term_memory': self.long_term_memory[batch_index],
            'history': self.history[batch_index],
            'nodes_list': self.nodes_list[batch_index],
            'graph': self.graph[batch_index],
            'trajectory': self.trajectory[batch_index],
            'planning': self.planning[batch_index],
        }

    async def generate_graph_description(self, batch_index: int) -> str:
        graph = self.graph[batch_index]
        prompt = f"Given the following graph structure:\n{graph}\n"
        prompt += "Generate a concise linguistic description of this graph, focusing on the connections between nodes and any notable patterns or structures."

        user_content = []
        user_content.append(
        {
            "type": "text",
            "text": prompt
        })
        system = """You are an embodied robot that navigates in the real world."""
        description = await self.generate_with_llm(system=system, user_content=user_content)
        return description

    async def analyze_trajectory(self, batch_index: int) -> str:
        trajectory = self.trajectory[batch_index]
        prompt = f"Given the following navigation trajectory:\n{trajectory}\n"
        prompt += "Analyze this trajectory and provide insights such as:\n"
        prompt += "1. Any patterns or repetitions in the path\n"
        prompt += "2. Potential areas of exploration that have been missed\n"
        prompt += "3. Suggestions for optimizing the path"

        user_content = []
        user_content.append(
        {
            "type": "text",
            "text": prompt
        })
        system = """You are an embodied robot that navigates in the real world."""
        analysis = await self.generate_with_llm(system=system, user_content=user_content)

        return analysis

    async def query_long_term_memory(self, batch_index: int, query: str) -> str:
        long_term_data = self.long_term_memory[batch_index]
        prompt = f"Given the following long-term memory data:\n{long_term_data}\n"
        prompt += f"Answer the following query: {query}"

        user_content = []
        user_content.append(
        {
            "type": "text",
            "text": prompt
        })
        system = """You are an embodied robot that navigates in the real world."""
        response = await self.generate_with_llm(system=system, user_content=user_content)
    
        return response
#code to debug  
from typing import List, Dict, Any
import math

class ActionAgent:
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
        next_action = self._parse_plan(plan)
        return self._match_action(next_action, action_options, t)

    def _parse_plan(self, plan: str) -> str:
        """
        Extract the next intended action from the plan.
        """
        sentences = plan.split('.')
        return sentences[0].strip().lower() if sentences else ""

    def _match_action(self, intended_action: str, action_options: List[str], t: int) -> int:
        """
        Match the intended action with available options or choose the best alternative.
        """
        action_scores = [self._calculate_action_score(intended_action, option, i, t) 
                         for i, option in enumerate(action_options)]
        return max(range(len(action_scores)), key=action_scores.__getitem__)

    def _calculate_action_score(self, intended_action: str, option: str, index: int, t: int) -> float:
        """
        Calculate a score for how well an action option matches the intended action.
        """
        base_score = self._action_similarity(intended_action, option.lower())
        
        # Penalize 'stop' action in the first few steps
        if "stop" in option.lower() and t < self.args.stop_after:
            base_score -= 1000  # Large penalty to avoid selecting 'stop'
        
        # Slight preference for continuing forward if scores are tied
        if "forward" in option.lower():
            base_score += 0.1
        
        return base_score

    def _action_similarity(self, intended_action: str, option: str) -> float:
        """
        Calculate similarity between intended action and option.
        """
        action_phrases = {
            "go forward": ["go forward", "move forward", "continue straight"],
            "turn left": ["turn left", "go left"],
            "turn right": ["turn right", "go right"],
            "turn around": ["turn around", "go back"],
            "go up": ["go up", "move up", "ascend"],
            "go down": ["go down", "move down", "descend"],
            "stop": ["stop", "halt", "end navigation"]
        }

        max_similarity = 0
        for action, phrases in action_phrases.items():
            if action in option:
                for phrase in phrases:
                    similarity = self._phrase_similarity(intended_action, phrase)
                    max_similarity = max(max_similarity, similarity)

        return max_similarity

    def _phrase_similarity(self, phrase1: str, phrase2: str) -> float:
        """
        Calculate a simple similarity score between two phrases.
        """
        words1 = set(phrase1.split())
        words2 = set(phrase2.split())
        return len(words1.intersection(words2)) / max(len(words1), len(words2))

    def get_action_concept(self, rel_heading, rel_elevation):
        """
        Get the action concept based on relative heading and elevation.
        """
        if rel_elevation > 0:
            return 'go up'
        elif rel_elevation < 0:
            return 'go down'
        else:
            if rel_heading == 0:
                return 'go forward'
            elif -math.pi / 2 <= rel_heading < 0:
                return 'turn left'
            elif 0 < rel_heading <= math.pi / 2:
                return 'turn right'
            else:
                return 'turn around'
                
class PlanningAgent(Agent):
    def __init__(self, name: str, llm_client: OpenAI):
        super().__init__(name, llm_client)
        self.current_plan = []

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a long-term plan based on the instruction, linguistic graph representation, and current position.

        :param input_data: A dictionary containing instruction, graph_description, and current_position
        :return: A dictionary containing the generated plan and any additional planning information
        """
        instruction = input_data.get('instruction', '')
        graph_description = input_data.get('graph_description', '')
        current_position = input_data.get('current_position', '')
        perception_data = input_data.get('perception_data', {})

        await self.log(f"Generating plan for instruction: {instruction}")

        # Generate a plan using the LLM
        plan = await self._generate_plan(instruction, graph_description, current_position, perception_data)

        # Update the current plan
        self.current_plan = plan

        return {
            'plan': plan,
            'plan_summary': await self._summarize_plan(plan),
            'estimated_steps': len(plan)
        }

    async def _generate_plan(self, instruction: str, graph_description: str, current_position: str, perception_data: Dict[str, Any]) -> List[str]:
        """
        Use the LLM to generate a detailed plan based on the given inputs.

        :param instruction: The navigation instruction
        :param graph_description: The linguistic description of the graph
        :param current_position: The current position in the environment
        :param perception_data: The latest perception data from the environment
        :return: A list of plan steps
        """
        prompt = f"""
        Given the following information, generate a detailed step-by-step plan to navigate to the destination:

        Instruction: {instruction}

        Current Position: {current_position}

        Environment Description:
        {graph_description}

        Current Perception:
        {self._format_perception_data(perception_data)}

        Generate a plan with the following considerations:
        1. Break down the navigation into clear, actionable steps.
        2. Include specific directions (e.g., "turn left", "go straight", "take the second right").
        3. Reference landmarks or notable features mentioned in the environment description or perception data.
        4. Estimate distances for each step if possible.
        5. Include any necessary decision points or alternative routes.

        Provide the plan as a numbered list of steps.
        """
        user_content = []
        user_content.append(
        {
            "type": "text",
            "text": prompt
        })
        system = """You are an embodied robot that navigates in the real world."""
        response = await self.generate_with_llm(system=system, user_content=user_content)
        plan_steps = self._parse_plan_steps(response)

        return plan_steps

    def _format_perception_data(self, perception_data: Dict[str, Any]) -> str:
        """Format perception data for inclusion in the LLM prompt."""
        current_view = perception_data.get('current_view', {})
        formatted_data = f"""
        Scene Description: {current_view.get('scene_description', 'N/A')}
        Key Objects: {', '.join(current_view.get('key_objects', []))}
        Landmarks: {', '.join(current_view.get('landmarks', []))}
        Visible Text: {current_view.get('visible_text', 'N/A')}
        Estimated Distances: {current_view.get('distances', 'N/A')}
        Navigation Info: {current_view.get('navigation_info', 'N/A')}
        """
        return formatted_data

    def _parse_plan_steps(self, response: str) -> List[str]:
        """Parse the LLM response into a list of plan steps."""
        lines = response.strip().split('\n')
        steps = []
        for line in lines:
            if line.strip() and any(line.strip().startswith(str(i)) for i in range(1, 100)):
                steps.append(line.strip())
        return steps

    async def _summarize_plan(self, plan: List[str]) -> str:
        """
        Generate a brief summary of the plan using the LLM.

        :param plan: The list of plan steps
        :return: A summary of the plan
        """
        plan_text = "\n".join(plan)
        prompt = f"""
        Summarize the following navigation plan in 2-3 sentences:

        {plan_text}

        Summary:
        """

        user_content = []
        user_content.append(
        {
            "type": "text",
            "text": prompt
        })
        system = """You are an embodied robot that navigates in the real world."""
        summary = await self.generate_with_llm(system=system, user_content=user_content)
    
        return summary.strip()

    async def update_plan(self, new_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the current plan based on new information.

        :param new_data: New information that might affect the plan
        :return: The updated plan information
        """
        await self.log("Updating plan with new information")
        return await self.process(new_data)
    
# class CoordinatorAgent(Agent):
#     def __init__(self, name: str, llm_client: OpenAI, args: Any):
#         super().__init__(name, llm_client)
#         self.args = args
#         self.perception_agent = PerceptionAgent("Perception", llm_client)
#         self.map_agent = MapAgent("Map", llm_client)
#         self.planning_agent = PlanningAgent("Planning", llm_client)
#         self.action_agent = ActionAgent(args)
#         self.memory_agent = MemoryAgent("Memory", llm_client, args.batch_size, args.short_term_capacity)

#     async def process(self, obs: Dict[str, Any], cand_inputs: Dict[str, Any], t: int) -> int:
#         # Process perception
#         perception_result = await self.perception_agent.process(obs)

#         # Update map
#         map_update = await self.map_agent.process({
#             'obs': obs,
#             'cand_inputs': cand_inputs
#         })

#         # Update memory
#         memory_update = await self.memory_agent.process({
#             'batch_index': 0,  # Assuming single batch for simplicity
#             'action': obs.get('action', ''),
#             'new_node': obs['viewpoint'],
#             'new_node_img': obs.get('image'),
#             'connected_nodes': [c['viewpointId'] for c in obs.get('candidate', [])],
#             'new_plan': ''  # Will be updated later
#         })

#         # Generate or update plan
#         graph_description = await self.map_agent._generate_map_description()
#         plan = await self.planning_agent.process({
#             'instruction': obs['instruction'],
#             'graph_description': graph_description,
#             'current_position': obs['viewpoint'],
#             'perception_data': perception_result
#         })

#         # Update memory with new plan
#         await self.memory_agent.update_planning(0, plan['plan_summary'])

#         # Decide on action
#         action_index = await self.action_agent.process(
#             plan['plan'][0],  # Use the first step of the plan
#             cand_inputs['action_prompts'],
#             obs,
#             t
#         )

#         # Log the decision process
#         await self.log(f"Step {t}: Perception processed, map updated, plan generated, action decided: {cand_inputs['action_prompts'][action_index]}")

#         return action_index

#     async def initialize(self) -> None:
#         await super().initialize()
#         await asyncio.gather(
#             self.perception_agent.initialize(),
#             self.map_agent.initialize(),
#             self.planning_agent.initialize(),
#             self.memory_agent.initialize()
#         )

#     async def shutdown(self) -> None:
#         await asyncio.gather(
#             self.perception_agent.shutdown(),
#             self.map_agent.shutdown(),
#             self.planning_agent.shutdown(),
#             self.memory_agent.shutdown()
#         )
#         await super().shutdown()

# class MultiAgentNavigationSystem:
#     def __init__(self, args: Any):
#         self.args = args
#         self.llm_client = OpenAI(
#             base_url = 'http://localhost:11434/v1',
#             api_key='ollama',
#         )  
#         self.coordinator = CoordinatorAgent("Coordinator", self.llm_client, args)

#     async def navigate(self, obs: Dict[str, Any], cand_inputs: Dict[str, Any], t: int) -> int:
#         return await self.coordinator.process(obs, cand_inputs, t)

#     async def initialize(self) -> None:
#         await self.coordinator.initialize()

#     async def shutdown(self) -> None:
#         await self.coordinator.shutdown()
# # Usage
# async def main():
#     batch_size = 2
#     map_agent = MapAgent(batch_size)
    
#     # [{'instr_id': '6250_2', 'scan': 'VLzqgDo317F', 'viewpoint': 'af3af33b0120469c9a00daa0d0b36799', 'viewIndex': 19, 'position': (2.2407100200653076, -5.2129998207092285, 4.926660060882568), 'heading': 3.665191429188092, 'elevation': 0.0, 'candidate': [{'heading': -1.0016941434798468, 'elevation': -0.06440292794585596, 'normalized_heading': 2.663497285708245, 'normalized_elevation': -0.06440292794585596, 'scanId': 'VLzqgDo317F', 'viewpointId': 'c55fa077761c4154af4b26da88eee80d', 'pointId': 17, 'distance': 0.07885618074593771, 'idx': 1, 'position': (3.605950117111206, -7.847620010375977, 4.735290050506592), 'caption': None, 'image': '/media/mlr_lab/6E18DC183015F19C/Ashu/Ashutosh_Dataset/VLN/Docker_Base/MapGPT/RGB_Observations/VLzqgDo317F/af3af33b0120469c9a00daa0d0b36799/17.jpg', 'absolute_heading': 2.617993877991494, 'absolute_elevation': 0.0, 'pretrained_inference': None}, {'heading': -0.572767977002155, 'elevation': -0.07284135849417891, 'normalized_heading': 3.092423452185937, 'normalized_elevation': -0.07284135849417891, 'scanId': 'VLzqgDo317F', 'viewpointId': '5be145994f974347850a48cecd04cdcd', 'pointId': 18, 'distance': 0.08788329689975492, 'idx': 1, 'position': (2.366260051727295, -7.764369964599609, 4.740260124206543), 'caption': None, 'image': '/media/mlr_lab/6E18DC183015F19C/Ashu/Ashutosh_Dataset/VLN/Docker_Base/MapGPT/RGB_Observations/VLzqgDo317F/af3af33b0120469c9a00daa0d0b36799/18.jpg', 'absolute_heading': 3.141592653589793, 'absolute_elevation': 0.0, 'pretrained_inference': None}], 'navigableLocations': [<MatterSim.ViewPoint object at 0x7f7056616230>, <MatterSim.ViewPoint object at 0x7f7056616430>], 'instruction': 'walk forward then turn right at the stairs then go down the stairs.', 'instr_encoding': None, 'gt_path': ['af3af33b0120469c9a00daa0d0b36799', '5be145994f974347850a48cecd04cdcd', '79aedad1206b4eea9c4b639ea2182eb7', '1c91ed40af2246f2b126dd0f661970df', '385019f5d018430fa233d483b253076c', 'fd263d778b534f798d0e1ae48886e5f3'], 'path_id': 6250, 'surrounding_tags': None, 'distance': 11.65849160592811}]

#     obs = [
#         {
#             'viewpoint': 'A',
#             'candidate': [
#                 {'viewpointId': 'B', 'image': 'image_B'},
#                 {'viewpointId': 'C', 'image': 'image_C'}
#             ]
#         },
#         {
#             'viewpoint': 'X',
#             'candidate': [
#                 {'viewpointId': 'Y', 'image': 'image_Y'},
#                 {'viewpointId': 'Z', 'image': 'image_Z'}
#             ]
#         }
#     ]
#     cand_inputs = {}  # Add relevant candidate inputs if needed

#     map_data = await map_agent.process(obs, cand_inputs)
#     print("Updated Map Data:", map_data)

#     # Generate map prompts for each batch
#     for i in range(batch_size):
#         trajectory_text, graph_text, graph_supp_text = map_agent.make_map_prompt(i)
#         print(f"\nBatch {i}:")
#         print("Trajectory:", trajectory_text)
#         print("Graph:", graph_text)
#         print("Supplementary Info:", graph_supp_text)

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

# if __name__ == "__main__":
#     asyncio.run(main())