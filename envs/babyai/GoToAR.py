#!/usr/bin/env python3

from __future__ import annotations

from minigrid.envs.babyai.goto import GoTo
# from minigrid.wrappers import ImgObsWrapper,ReseedWrapper
from gym import spaces
from minigrid.envs.babyai.core.verifier import GoToInstr, ObjDesc

class GoToAR(GoTo):
    def __init__(self, tile_size=8, seeds=[0], seed_idx=0, objtype=None, objcolor=None,**kwargs):
        self.seeds = list(seeds)
        self.seed_idx = seed_idx
        self.objtype = objtype
        self.objcolor = objcolor
        super().__init__(**kwargs)
        self.tile_size = tile_size
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height * tile_size, self.width * tile_size, 3),
            dtype='uint8'
        )
        action_size = self.action_space.n
        self.action_space = spaces.Discrete(action_size)


    def step(self, action):
        """Returns a modified observation using :meth:`self.observation` after calling :meth:`env.step`."""
        obs, reward, terminated, truncated, info = super().step(action)
        done = terminated or truncated
        return self.observation(obs), reward, done, info

    def observation(self, obs):
        rgb_img = self.get_frame(highlight=False, tile_size=self.tile_size)

        return rgb_img

    def render(self,mode = "human", **kwargs):
        self.render_mode = mode
        return super().render()

    def seed(self, seed):
        pass

    def reset(self, **kwargs):
        """Resets the environment, returning a modified observation using :meth:`self.observation`."""
        seed = self.seeds[self.seed_idx]
        self.seed_idx = (self.seed_idx + 1) % len(self.seeds)
        obs,info = super().reset(seed=seed,options=None)
        return self.observation(obs)

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        self.check_objs_reachable()
        obj = self._rand_elem(objs)

        if self.objtype and self.objcolor:
            objtype = self.objtype
            objcolor = self.objcolor
        else:
            objtype = obj.type
            objcolor = obj.color

        self.instrs = GoToInstr(ObjDesc(objtype, objcolor))

        # If requested, open all the doors
        if self.doors_open:
            self.open_all_doors()
if __name__ == "__main__":
    print("okk")
