import numpy as np
import matplotlib.pyplot as plt
from fdmEnv import WVRAC

def simulate_maneuver(mode, steps=1200, psi_t_init=np.pi/3):
    env = WVRAC()
    env.reset()
    x_t_list, y_t_list = [], []
    psi_t = psi_t_init
    phi_t = 0
    v_t = 60
    env.psi_t_des = None
    env.s_turn_stage = None
    env.s_turn_psi_list = None
    env.s_turn_direction = None

    for _ in range(steps):
        # 只测试目标机动，不考虑追击方
        phi_t = env.maneuver_library(mode, psi_t)
        dpsi_t = 9.81 / v_t * np.tan(phi_t)
        psi_t += dpsi_t * env.dt
        psi_t = env.check_heading(psi_t)
        if hasattr(env, 'state'):
            x_t = env.state['x_t']
            y_t = env.state['y_t']
        else:
            x_t, y_t = 0, 0
        dx_t = v_t * np.sin(psi_t)
        dy_t = v_t * np.cos(psi_t)
        x_t += dx_t * env.dt
        y_t += dy_t * env.dt
        x_t_list.append(x_t)
        y_t_list.append(y_t)
        
        env.state['x_t'] = x_t
        env.state['y_t'] = y_t
        env.state['psi_t'] = psi_t
        env.state['phi_t'] = phi_t

    return x_t_list, y_t_list

if __name__ == "__main__":
    plt.figure(figsize=(8, 8))
    for mode, label in zip([1, 2], ["single", "S-turn"]):
        x, y = simulate_maneuver(mode)
        plt.plot(x, y, label=label)
    plt.xlabel("x_t")
    plt.ylabel("y_t")
    plt.title("traj")
    plt.legend()
    plt.axis('equal')
    plt.grid()
    # plt.show()
    plt.savefig("maneuver_test.png", dpi=300)
    plt.close()