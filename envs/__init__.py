from gym.envs.registration import register

register(
    id='pursuitevasion-v0',
    entry_point='envs.pursuit_evasion3d.pursuit_evasion3d:PursuitEvasion3d',
    max_episode_steps=200,
    kwargs={}
)

register(
    id='buparimage-v0',
    entry_point='envs.gym_minigrid.buparimage:buparimage',
    max_episode_steps=100000,
    kwargs={}
)

register(
    id='bupfullgrid-v0',
    entry_point='envs.gym_minigrid.bupfullgrid:bupfullgrid',
    max_episode_steps=100000,
    kwargs={}
)

register(
    id='rooms-v0',
    entry_point='envs.rooms.rooms:RoomsEnv',
    max_episode_steps=100000,
    kwargs={}
)

register(
    id='unlockar-v0',
    entry_point='envs.gym_minigrid.unlockar:unlockar',
    max_episode_steps=100000,
    kwargs={}
)

register(
    id='memoryar-v0',
    entry_point='envs.gym_minigrid.memory:memoryar',
    max_episode_steps=100000,
    kwargs={}
)

register(
    id='unlockpickupar-v0',
    entry_point='envs.gym_minigrid.unlockpickupar:unlockpickupar',
    max_episode_steps=100000,
    kwargs={}
)

register(
    id='unlockpickupactionbonus-v0',
    entry_point='envs.gym_minigrid.unlockpickupactionbonus:unlockpickupactionbonus',
    max_episode_steps=100000,
    kwargs={}
)

register(
    id='unlockpickupnoisear-v0',
    entry_point='envs.gym_minigrid.unlockpickupnoisear:unlockpickupnoisear',
    max_episode_steps=100000,
    kwargs={}
)

register(
    id='unlockpickupnoiseactionbonus-v0',
    entry_point='envs.gym_minigrid.unlockpickupnoiseactionbonus:unlockpickupnoiseactionbonus',
    max_episode_steps=100000,
    kwargs={}
)

register(
    id='unlockpickupuncertaingoals-v0',
    entry_point='envs.gym_minigrid.unlockpickupUncertainGoals:unlockpickupUncertainGoals',
    max_episode_steps=100000,
    kwargs={}
)

register(
    id='unlockpickupuncertaingoalsactionbonus-v0',
    entry_point='envs.gym_minigrid.unlockpickupUncertainGoalsactionbonus:unlockpickupUncertainGoalsactionbouns',
    max_episode_steps=100000,
    kwargs={}
)

register(
    id='girdCL-v0',
    entry_point='envs.ToyMaze.Gridworld_CL:Gridworld_CL',
    max_episode_steps=100000,
    kwargs={}
)

register(
    id='ObstructedMaze_1DlhAR-v0',
    entry_point='envs.gym_minigrid.obstructedmazear:ObstructedMaze_1DlhAR',
    max_episode_steps=100000,
    kwargs={}
)

register(
    id='ObstructedMaze_2DlhbAR-v0',
    entry_point='envs.gym_minigrid.obstructedmazear:ObstructedMaze_2DlhbAR',
    max_episode_steps=100000,
    kwargs={}
)
register(
    id='ObstructedMaze_FullAR-v0',
    entry_point='envs.gym_minigrid.obstructedmazear:ObstructedMaze_FullAR',
    max_episode_steps=100000,
    kwargs={}
)
register(
    id='ObstructedMaze_2DlhAR-v0',
    entry_point='envs.gym_minigrid.obstructedmazear:ObstructedMaze_2DlhAR',
    max_episode_steps=100000,
    kwargs={}
)
register(
    id='ObstructedMaze_1QAR-v0',
    entry_point='envs.gym_minigrid.obstructedmazear:ObstructedMaze_1QAR',
    max_episode_steps=100000,
    kwargs={}
)

register(
    id='ObstructedMaze_1DlAR-v0',
    entry_point='envs.gym_minigrid.obstructedmazear:ObstructedMaze_1DlAR',
    max_episode_steps=100000,
    kwargs={}
)

register(
    id='ObstructedMaze_1DlhbAR-v0',
    entry_point='envs.gym_minigrid.obstructedmazear:ObstructedMaze_1DlhbAR',
    max_episode_steps=100000,
    kwargs={}
)

register(
    id='ObstructedMaze_2QAR-v0',
    entry_point='envs.gym_minigrid.obstructedmazear:ObstructedMaze_2QAR',
    max_episode_steps=100000,
    kwargs={}
)
register(
    id='ObstructedMaze_2DlAR-v0',
    entry_point='envs.gym_minigrid.obstructedmazear:ObstructedMaze_2DlAR',
    max_episode_steps=100000,
    kwargs={}
)

register(
    id='UnlockToUnlockAR-v0',
    entry_point='envs.babyai.unlocktounlockar:UnlockToUnlockAR',
    max_episode_steps=100000,
    kwargs={}
)

register(
    id='UnlockToUnlockPositionBonus-v0',
    entry_point='envs.babyai.unlocktounlockposbonus:UnlockToUnlockPositionBonus',
    max_episode_steps=100000,
    kwargs={}
)

register(
    id='GoToRedBallGreyAR-v0',
    entry_point='envs.babyai.GoToRedBallGreyAR:GoToRedBallGreyAR',
    max_episode_steps=100000,
    kwargs={}
)

register(
    id='GoToRedBallGreyPositionBonus-v0',
    entry_point='envs.babyai.GoToRedBallGreyposbonus:GoToRedBallGreyPositionBonus',
    max_episode_steps=100000,
    kwargs={}
)

# GoTo: defalut:3*3
GoToSeed = [0]
register(
    id='GoToAR-v0',
    entry_point='envs.babyai.GoToAR:GoToAR',
    max_episode_steps=100000,
    kwargs={"seeds":GoToSeed}
)

register(
    id='GoToPositionBonus-v0',
    entry_point='envs.babyai.GoToposbonus:GoToPositionBonus',
    max_episode_steps=100000,
    kwargs={"seeds":GoToSeed}
)

register(
    id='GoToAddPositionBonus-v0',
    entry_point='envs.babyai.GoToaddposbonus:GoToAddPositionBonus',
    max_episode_steps=100000,
    kwargs={"seeds":GoToSeed}
)

target_list = {"BlueBall":{"color":"blue","type":"ball"},
               "PurpleBall":{"color":"purple","type":"ball"},
               "BlueKey":{"color":"blue","type":"key"},
               "GreyKey":{"color":"grey","type":"key"},
               "GreenBox":{"color":"green","type":"box"},
               "RedBox":{"color":"red","type":"box"},}

for name,setting in target_list.items():
    color = setting["color"]
    ttype = setting["type"]

    register(
        id='GoToR3'+name+'AR-v0',
        entry_point='envs.babyai.GoToAR:GoToAR',
        max_episode_steps=100000,
        kwargs={"seeds": GoToSeed,"objcolor":color, "objtype":ttype}
    )

    register(
        id='GoToR3'+name+'AddPositionBonus-v0',
        entry_point='envs.babyai.GoToaddposbonus:GoToAddPositionBonus',
        max_episode_steps=100000,
        kwargs={"seeds": GoToSeed,"objcolor":color, "objtype":ttype}
    )



register(
    id='GoToR2AR-v0',
    entry_point='envs.babyai.GoToAR:GoToAR',
    max_episode_steps=100000,
    kwargs={"num_rows":2 ,"num_cols":2, "num_dists":16, "seeds":[5]}
)

register(
    id='GoToR2PositionBonus-v0',
    entry_point='envs.babyai.GoToposbonus:GoToPositionBonus',
    max_episode_steps=100000,
    kwargs={"num_rows":2 ,"num_cols":2, "num_dists":16, "seeds":[5]}
)

register(
    id='GoToDoorOpenR2AR-v0',
    entry_point='envs.babyai.GoToAR:GoToAR',
    max_episode_steps=100000,
    kwargs={"num_rows":2 ,"num_cols":2, "num_dists":16, "seeds":[5], "doors_open":True}
)

register(
    id='GoToDoorOpenR2PositionBonus-v0',
    entry_point='envs.babyai.GoToposbonus:GoToPositionBonus',
    max_episode_steps=100000,
    kwargs={"num_rows":2 ,"num_cols":2, "num_dists":16, "seeds":[5], "doors_open":True}
)

register(
    id='GoToDoorOpenR2AddPositionBonus-v0',
    entry_point='envs.babyai.GoToaddposbonus:GoToAddPositionBonus',
    max_episode_steps=100000,
    kwargs={"num_rows":2 ,"num_cols":2, "num_dists":16, "seeds":[5], "doors_open":True}
)

# this env is the same as GoToDoorOpenR2, with different goal
register(
    id='GoToDoorOpenR2GreyKeyAR-v0',
    entry_point='envs.babyai.GoToAR:GoToAR',
    max_episode_steps=100000,
    kwargs={"num_rows":2 ,"num_cols":2, "num_dists":16, "seeds":[5], "doors_open":True,
            "objcolor":"grey", "objtype":"key"}
)

register(
    id='GoToDoorOpenR2GreyKeyAddPositionBonus-v0',
    entry_point='envs.babyai.GoToaddposbonus:GoToAddPositionBonus',
    max_episode_steps=100000,
    kwargs={"num_rows":2 ,"num_cols":2, "num_dists":16, "seeds":[5], "doors_open":True,
            "objcolor":"grey", "objtype":"key"}
)

register(
    id='GoToDoorOpenR2GreenBoxAR-v0',
    entry_point='envs.babyai.GoToAR:GoToAR',
    max_episode_steps=100000,
    kwargs={"num_rows":2 ,"num_cols":2, "num_dists":16, "seeds":[5], "doors_open":True,
            "objcolor":"green", "objtype":"box"}
)

register(
    id='GoToDoorOpenR2GreenBoxAddPositionBonus-v0',
    entry_point='envs.babyai.GoToaddposbonus:GoToAddPositionBonus',
    max_episode_steps=100000,
    kwargs={"num_rows":2 ,"num_cols":2, "num_dists":16, "seeds":[5], "doors_open":True,
            "objcolor":"green", "objtype":"box"}
)



GoToRedBlueBallSeed= [5]
register(
    id='GoToRedBallAR-v0',
    entry_point='envs.babyai.GoToRedBlueBall.GoToRedBallAR:GoToRedBallAR',
    max_episode_steps=100000,
    kwargs={"seeds":GoToRedBlueBallSeed}
)

register(
    id='GoToRedBallPositionBonus-v0',
    entry_point='envs.babyai.GoToRedBlueBall.GoToRedBallposbonus:GoToRedBallPositionBonus',
    max_episode_steps=100000,
    kwargs={"seeds":GoToRedBlueBallSeed}
)


register(
    id='GoToBlueBallAR-v0',
    entry_point='envs.babyai.GoToRedBlueBall.GoToBlueBallAR:GoToBlueBallAR',
    max_episode_steps=100000,
    kwargs={"seeds":GoToRedBlueBallSeed}
)

GoToLocalSeed = [15]
register(
    id='GoToLocalAR-v0',
    entry_point='envs.babyai.GoToLocal.GoToLocalAR:GoToLocalAR',
    max_episode_steps=100000,
    kwargs={"seeds":GoToLocalSeed}
)

register(
    id='GoToLocalGreyBoxAR-v0',
    entry_point='envs.babyai.GoToLocal.GoToLocalAR:GoToLocalAR',
    max_episode_steps=100000,
    kwargs={"seeds":GoToLocalSeed, "objcolor":"grey", "objtype":"box"}
)

register(
    id='GoToLocalPurpleBallAR-v0',
    entry_point='envs.babyai.GoToLocal.GoToLocalAR:GoToLocalAR',
    max_episode_steps=100000,
    kwargs={"seeds":GoToLocalSeed, "objcolor":"purple", "objtype":"ball"}
)

register(
    id='GoToLocalYellowKeyAR-v0',
    entry_point='envs.babyai.GoToLocal.GoToLocalAR:GoToLocalAR',
    max_episode_steps=100000,
    kwargs={"seeds":GoToLocalSeed, "objcolor":"yellow", "objtype":"key"}
)

register(
    id='GoToLocalPositionBonus-v0',
    entry_point='envs.babyai.GoToLocal.GoToLocalposbonus:GoToLocalPositionBonus',
    max_episode_steps=100000,
    kwargs={"seeds":GoToLocalSeed}
)

register(
    id='MiniBossLevelAR-v0',
    entry_point='envs.babyai.BossLevel.MiniBossLevelAR:MiniBossLevelAR',
    max_episode_steps=100000,
    kwargs={}
)

register(
    id='MiniBossLevelPositionBonus-v0',
    entry_point='envs.babyai.BossLevel.MiniBossLevelposbonus:MiniBossLevelPositionBonus',
    max_episode_steps=100000,
    kwargs={}
)

register(
    id='MiniBossLevelAddPositionBonus-v0',
    entry_point='envs.babyai.BossLevel.MiniBossLeveladdposbonus:MiniBossLevelAddPositionBonus',
    max_episode_steps=100000,
    kwargs={}
)

register(
    id='BossLevelAR-v0',
    entry_point='envs.babyai.BossLevel.BossLevelAR:BossLevelAR',
    max_episode_steps=100000,
    kwargs={}
)

register(
    id='BossLevelPositionBonus-v0',
    entry_point='envs.babyai.BossLevel.BossLevelposbonus:BossLevelPositionBonus',
    max_episode_steps=100000,
    kwargs={}
)

register(
    id='BossLevelAddPositionBonus-v0',
    entry_point='envs.babyai.BossLevel.BossLeveladdposbonus:BossLevelAddPositionBonus',
    max_episode_steps=100000,
    kwargs={}
)

register(
    id='BossLevelNoUnlockAR-v0',
    entry_point='envs.babyai.BossLevel.BossLevelNoUnlockAR:BossLevelNoUnlockAR',
    max_episode_steps=100000,
    kwargs={}
)

register(
    id='BossLevelNoUnlockPositionBonus-v0',
    entry_point='envs.babyai.BossLevel.BossLevelNoUnlockposbonus:BossLevelNoUnlockPositionBonus',
    max_episode_steps=100000,
    kwargs={}
)

register(
    id='BossLevelNoUnlockAddPositionBonus-v0',
    entry_point='envs.babyai.BossLevel.BossLevelNoUnlockaddposbonus:BossLevelNoUnlockAddPositionBonus',
    max_episode_steps=100000,
    kwargs={}
)


###KeyCorridor##
KeyCorridorS4R2Kwargs = {"room_size": 4, "num_rows": 2}
register(
    id='KeyCorridorS4R2AR-v0',
    entry_point='envs.babyai.KeyCorridor.KeyCorridorAR:KeyCorridorAR',
    max_episode_steps=100000,
    kwargs=KeyCorridorS4R2Kwargs
)

register(
    id='KeyCorridorS4R2PositionBonus-v0',
    entry_point='envs.babyai.KeyCorridor.KeyCorridorposbonus:KeyCorridorPositionBonus',
    max_episode_steps=100000,
    kwargs=KeyCorridorS4R2Kwargs
)

register(
    id='KeyCorridorS4R2AddPositionBonus-v0',
    entry_point='envs.babyai.KeyCorridor.KeyCorridoraddposbonus:KeyCorridorAddPositionBonus',
    max_episode_steps=100000,
    kwargs=KeyCorridorS4R2Kwargs
)

KeyCorridorS5R2Kwargs = {"room_size": 5, "num_rows": 2}
register(
    id='KeyCorridorS5R2AR-v0',
    entry_point='envs.babyai.KeyCorridor.KeyCorridorAR:KeyCorridorAR',
    max_episode_steps=100000,
    kwargs=KeyCorridorS5R2Kwargs
)

register(
    id='KeyCorridorS5R2PositionBonus-v0',
    entry_point='envs.babyai.KeyCorridor.KeyCorridorposbonus:KeyCorridorPositionBonus',
    max_episode_steps=100000,
    kwargs=KeyCorridorS5R2Kwargs
)

register(
    id='KeyCorridorSS5R2AddPositionBonus-v0',
    entry_point='envs.babyai.KeyCorridor.KeyCorridoraddposbonus:KeyCorridorAddPositionBonus',
    max_episode_steps=100000,
    kwargs=KeyCorridorS5R2Kwargs
)

#PutNext
PutNextLocalSeed = [10]
register(
    id='PutNextLocalAR-v0',
    entry_point='envs.babyai.PutNextLocal.PutNextLocalAR:PutNextLocalAR',
    max_episode_steps=100000,
    kwargs={"seeds":PutNextLocalSeed}
)

register(
    id='PutNextLocalPositionBonus-v0',
    entry_point='envs.babyai.PutNextLocal.PutNextLocalposbonus:PutNextLocalPositionBonus',
    max_episode_steps=100000,
    kwargs={"seeds":PutNextLocalSeed}
    )





register(
    id='GoToDoorOpenR2RedBallAR-v0',
    entry_point='envs.babyai.GoToAR:GoToAR',
    max_episode_steps=100000,
    kwargs={"num_rows":2 ,"num_cols":2, "num_dists":16, "seeds":[5], "doors_open":True,
            "objcolor":"red", "objtype":"ball"}
)

register(
    id='GoToDoorOpenR2RedBallAddPositionBonus-v0',
    entry_point='envs.babyai.GoToaddposbonus:GoToAddPositionBonus',
    max_episode_steps=100000,
    kwargs={"num_rows":2 ,"num_cols":2, "num_dists":16, "seeds":[5], "doors_open":True,
            "objcolor":"red", "objtype":"ball"}
)

register(
    id='GoToDoorOpenR2PurpleBallAddPositionBonus-v0',
    entry_point='envs.babyai.GoToaddposbonus:GoToAddPositionBonus',
    max_episode_steps=100000,
    kwargs={"num_rows":2 ,"num_cols":2, "num_dists":16, "seeds":[5], "doors_open":True,
            "objcolor":"purple", "objtype":"ball"}
)

register(
    id='GoToDoorOpenR2BlueBallAR-v0',
    entry_point='envs.babyai.GoToAR:GoToAR',
    max_episode_steps=100000,
    kwargs={"num_rows":2 ,"num_cols":2, "num_dists":16, "seeds":[5], "doors_open":True,
            "objcolor":"blue", "objtype":"ball"}
)

register(
    id='GoToDoorOpenR2BlueBallAddPositionBonus-v0',
    entry_point='envs.babyai.GoToaddposbonus:GoToAddPositionBonus',
    max_episode_steps=100000,
    kwargs={"num_rows":2 ,"num_cols":2, "num_dists":16, "seeds":[5], "doors_open":True,
            "objcolor":"blue", "objtype":"ball"}
)

register(
    id='GoToDoorOpenR2GreenBallAR-v0',
    entry_point='envs.babyai.GoToAR:GoToAR',
    max_episode_steps=100000,
    kwargs={"num_rows":2 ,"num_cols":2, "num_dists":16, "seeds":[5], "doors_open":True,
            "objcolor":"green", "objtype":"ball"}
)

#
# # atari
# register(
#     id='ALE/Breakout-v5',
#     entry_point='envs.Atari.atari:Atari_Env',
#     kwargs={"env_id":"ALE/Breakout-v5","render_mode":"human","env_seed":1,"obs_type":"grayscale",
#                             "frameskip":4}
# )