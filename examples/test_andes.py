import andes
# '/Users/yichenzhang/GitHub/andes_gym/andes_gym/envs/ieee14_alter_pq.xlsx'
sim_case = andes.run("../andes_gym/envs/ieee14_alter_pq.xlsx", no_output=True)
sim_case.PQ.config.p2p = 1
sim_case.PQ.config.p2z = 0
sim_case.PQ.config.p2i = 0
sim_case.PQ.config.q2q = 1
sim_case.PQ.config.q2z = 0
sim_case.PQ.config.q2i = 0
sim_case.TDS.init()
sim_case.TDS.config.fixt = True