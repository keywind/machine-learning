import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import collections

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        state_fields = ['next_waypoint', 'light']
        self.state_tuple = collections.namedtuple('State', state_fields)
        self.q_table = {}
        self.alpha = 0.2 # learning rate 
        self.gamma = 0.8 # discount factor
        self.epsilon = 0.8 # exploration probability 

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = self.state_tuple(self.next_waypoint, inputs['light'])

        # TODO: Select action according to your policy
        # choose either explore or exploit
        if random.uniform(0, 1) < self.epsilon :
            action = self.max_action_for_state(self.state)
        else:  
            action = random.choice([None, 'forward','left','right'])
        
        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        self.next_waypoint = self.planner.next_waypoint()
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        next_state = self.state_tuple(self.next_waypoint, inputs['light'])

        # Update Q Table
        if (self.state, action) not in self.q_table:
            self.q_table[self.state, action] = 0
        self.q_table[self.state, action] = (1 - self.alpha) * self.q_table[self.state, action] + self.alpha * (reward + self.gamma * self.max_value_for_state(next_state))

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

    def max_action_for_state(self, s):
        max_action = None
        max_value = float("-inf")
        actions = [None, 'forward','left','right']
        random.shuffle(actions)
        for action in [None, 'forward','left','right']:
            if (s,action) not in self.q_table:
                self.q_table[s, action] = 0
            if self.q_table[s, action] > max_value:
                max_value = self.q_table[s, action]
                max_action = action
        return max_action

    def max_value_for_state(self, s):
        max_value = float("-inf") 
        for action in [None, 'forward','left','right']:
            if (s,action) not in self.q_table:
                self.q_table[s, action] = 0
            if self.q_table[s, action] > max_value:
                max_value = self.q_table[s, action]
        return max_value


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
