import os
import sys
import time
import numpy as np
from mpi4py import MPI
import argparse
from FFAG_Bunch import FFAG_ManageBunchAttribute
from FFAG_ParasAndConversion import FFAG_ConversionTools
from FFAG_MathTools import fast_mod_njit, detect_azimuth_crossing, fast_mod_parallel, detect_azimuth_crossing_portion


# class StepDump:
#     def __init__(self, start_time, end_time, time_interval,
#                  tracked_particle_id_global: list[int],
#                  save_folder="dump_data"):
#         """
#         StepDump 类，用于在 RK 积分过程中每隔一定的 step 进行一次粒子 Dump 操作。
#         :param step_interval: int, 每隔多少步进行一次 Dump。
#         :param num_particles_to_dump: int, 每次 Dump 保存前 n 个粒子的状态。
#         :param save_folder: str, 保存文件的文件夹路径，不包含 rank，rank 号会自动添加为子文件夹。
#         """
#         self.start_time = start_time
#         self.end_time = end_time
#         self.time_interval = time_interval
#
#         self.save_folder = save_folder
#
#         # 获取 MPI 信息
#         comm = MPI.COMM_WORLD
#         rank = comm.Get_rank()
#         size = comm.Get_size()
#
#         # 创建保存路径，如果文件夹不存在则创建, 若文件夹已存在且不为空，则终止程序且给出提示
#         # 检查文件夹是否存在
#         if rank == 0:
#             if not os.path.exists(save_folder):
#                 os.makedirs(save_folder)
#                 print(f"文件夹 {save_folder} 已成功创建。")
#             else:
#                 # 如果文件夹存在，检查是否为空
#                 if os.listdir(save_folder):  # os.listdir() 返回文件夹中的文件列表
#                     sys.stderr.write(f"Error: 数据文件夹 {save_folder} 已存在且不为空，所有进程将被终止。\n")
#                     comm.Abort(1)  # 终止所有进程
#                 else:
#                     print(f"文件夹 {save_folder} 已成功创建。")
#             time.sleep(2)
#         # 确保所有进程都同步等 rank 0 完成检查
#         comm.Barrier()
#         # 在初始化时确定需要保存的随机粒子索引，后续使用相同的粒子
#         self.selected_indices = None
#
#         self.tracked_global_ids = np.asarray(tracked_particle_id_global, dtype=np.int64)
#         assert self.tracked_global_ids.ndim == 1, "tracked_particle_id_global 必须是 1D ndarray"
#         self.save_flag = False
#
#         self.particle_buffer_dict = {}  # gid → ndarray(max_records, dim)
#         self.cursor_dict = {}  # gid → 当前写入位置
#         self.write_done_dict = {}  # gid → True if 满了
#         self.local_tracked_ids = np.zeros((0,))  # 本地需要记录的粒子行号
#
#
#     def check_and_dump(self, time_Pre, time_Post, bunch_obj, data_type):
#         """
#         每个时间步调用一次，记录所有发生穿越的目标粒子状态。
#         """
#         if self.save_flag:
#             return
#
#         # 初始化本地需要保存的粒子ID
#         if np.size(self.local_tracked_ids,0)<1:
#             self.determine_local_ids(bunch_obj)
#             if np.size(self.local_tracked_ids,0)<1:
#                 return
#
#         # 追加保存缓冲区所有粒子数据
#         if not (self.start_time < time_Pre < self.end_time):
#             if time_Pre > self.end_time and not self.save_flag:
#                 self.dump_all_particles() # 需要完成代码
#                 self.save_flag = True
#             return
#
#         # 仅当 time_Post 处于可 Dump 的区间内才进行后续判断
#         if self.start_time < time_Post < self.end_time:
#             # 计算 time_Pre, time_Post 分别对应的索引
#             time_Pre_idx = (time_Pre - self.start_time) // self.time_interval
#             time_Post_idx = (time_Post - self.start_time) // self.time_interval
#
#             if time_Pre_idx == time_Post_idx:
#                 # 没有跨过新的 Dump 时刻，直接返回
#                 return
#             else:
#                 # 说明 (time_Pre, time_Post) 区间至少跨过一个 Dump 时刻
#                 # 往缓冲区存入粒子信息
#                 self.save_all_particles() # 需要完成代码
#         else:
#             return
#
#     def determine_local_ids(self, bunch_obj):
#         """
#         构建全局 ID 到本地行号的映射，找出本进程需记录的粒子。
#         """
#         global_ids_local = bunch_obj.LocalBunch[:, -1].astype(np.int64)
#         gid_to_row = {gid: row for row, gid in enumerate(global_ids_local)}
#         self.local_tracked_ids = [gid_to_row[gid] for gid in self.tracked_global_ids if gid in gid_to_row]
#         # print(f"[Rank {self.rank}] 本地匹配行号: {self.local_tracked_ids}")
#         self.local_tracked_ids=np.array(self.local_tracked_ids)
#         return self.local_tracked_ids
#
#     def dump_particles(self, bunch_obj, time_i, data_type):
#         """
#         执行粒子的 Dump 操作，保存已选定的 n 个粒子的信息。
#         :param bunch_obj: FFAG_Bunch 对象。
#         :param time_i: int, 当前时间或步数，用于标记当前的粒子状态。
#         """
#         # 确保已初始化粒子索引
#         if self.selected_indices is None:
#             self.initialize_particles_to_dump(bunch_obj)
#
#         # 根据选定的索引保存对应粒子的信息
#         particles_to_dump_v = bunch_obj.LocalBunch[self.selected_indices, :]
#         if data_type == "RK4":
#             particles_to_dump = FFAG_ConversionTools().ConvertVrzek2Przek(particles_to_dump_v)
#         elif data_type == "Boris":
#             particles_to_dump = FFAG_ConversionTools().ConvertVrzek2Przek_boris(particles_to_dump_v)
#         else:
#             raise ValueError("data_type 的值必须是 RK4 或 Boris")
#
#         # 获取当前 MPI 线程号
#         comm = MPI.COMM_WORLD
#         rank = comm.Get_rank()
#
#         # 遍历要保存的粒子，并将其信息追加到对应文件中
#         for i, particle in enumerate(particles_to_dump):
#             # 获取粒子的 Global_ID，作为文件名的一部分
#             global_id = int(particle[bunch_obj.BunchAttribute.Attribute['Global_ID']])
#
#             file_name = os.path.join(self.save_folder, f"particle_{global_id}_rank_{rank}.npz")
#
#             # 将粒子在当前步的状态追加写入文件
#             append_to_npz(file_name, np.concatenate((particle, [time_i])))


class StepDump:
    """
    简单逻辑：
    - 每个 step 给定 time_Pre(步前) 与 time_Post(步后)。
    - 若 (time_Pre, time_Post) 跨过了按 start_time + k*time_interval 划分的栅格，则把本步目标粒子状态追加到缓冲；
      追加的时间写入 time_Post（不插值）。
    - 当时间超过 end_time 时，落盘并标记完成（每粒子一个 .npz 文件，按 rank 区分）。
    说明：保持“开区间”语义（start_time < t < end_time），与您原代码一致。
    """

    def __init__(self, start_time, end_time, time_interval,
                 tracked_particle_id_global: list[int],
                 save_folder="dump_data"):
        """
        :param start_time: float, 记录起始时间
        :param end_time: float, 记录结束时间
        :param time_interval: float, 时间栅格间隔
        :param tracked_particle_id_global: list[int], 需要记录的全局粒子ID
        :param save_folder: str, 保存目录
        """
        self.start_time = float(start_time)
        self.end_time = float(end_time)
        self.time_interval = float(time_interval)
        assert self.time_interval > 0.0, "time_interval 必须 > 0"

        self.save_folder = save_folder

        # MPI
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # 目录检查（rank 0 负责）；非空则直接终止全部进程
        if self.rank == 0:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
                print(f"[StepDump] 目录 {save_folder} 已创建。")
            else:
                if os.listdir(save_folder):
                    sys.stderr.write(f"Error: 数据目录 {save_folder} 不为空，终止所有进程。\n")
                    self.comm.Abort(1)
                else:
                    print(f"[StepDump] 目录 {save_folder} 已存在（空）。")
            time.sleep(0.3)
        self.comm.Barrier()

        # 追踪的全局ID
        self.tracked_global_ids = np.asarray(tracked_particle_id_global, dtype=np.int64)
        assert self.tracked_global_ids.ndim == 1, "tracked_particle_id_global 必须是 1D"

        # 状态与缓冲
        self.save_flag = False
        self.local_tracked_ids = np.zeros((0,), dtype=np.int64)  # 本 rank 需要记录的本地行号
        self.particle_buffer_dict: dict[int, list] = {}          # gid → list[np.ndarray(row,)]

        # 转换器
        self._conv = FFAG_ConversionTools()

    def determine_local_ids(self, bunch_obj):
        """
        从 LocalBunch 最后一列读取全局ID，构建 gid->row 映射，找出本进程需记录的粒子行号。
        """
        global_ids_local = bunch_obj.LocalBunch[:, -1].astype(np.int64)
        gid_to_row = {gid: row for row, gid in enumerate(global_ids_local)}
        local_ids = [gid_to_row[gid] for gid in self.tracked_global_ids if gid in gid_to_row]
        self.local_tracked_ids = np.asarray(local_ids, dtype=np.int64)
        return self.local_tracked_ids

    def save_all_particles(self, bunch_obj, data_type, time_stamp):
        """
        仅对本 rank 上的目标粒子进行记录；若存在 Step_SurviveFlag，则仅记录存活的粒子。
        记录格式：ConvertedPrzek + time(秒)，时间列放最后。
        """
        if self.local_tracked_ids.size == 0:
            return

        survive_mask = getattr(bunch_obj, "Step_SurviveFlag", None)

        for local_id in self.local_tracked_ids:
            if survive_mask is not None and not survive_mask[local_id]:
                continue

            state = bunch_obj.LocalBunch[local_id:local_id + 1, :]  # shape (1, D)

            if data_type == "RK4":
                converted = self._conv.ConvertVrzek2Przek(state)
            elif data_type == "Boris":
                converted = self._conv.ConvertVrzek2Przek_boris(state)
            else:
                raise ValueError("data_type 必须为 'RK4' 或 'Boris'")

            tcol = np.array([[time_stamp]], dtype=np.float64)
            row = np.hstack((converted, tcol)).astype(np.float64).ravel()

            gid = int(bunch_obj.LocalBunch[local_id, -1])
            if gid not in self.particle_buffer_dict:
                self.particle_buffer_dict[gid] = []
            self.particle_buffer_dict[gid].append(row)

    def check_and_dump(self, time_Pre, time_Post, bunch_obj, data_type):
        """
        每个时间步调用一次：
        1) 若 (time_Pre, time_Post) 跨越了新的时间栅格（按 floor 索引判断），则把“本步状态”追加到缓冲（时间写 time_Post）；
        2) 若 time_Pre > end_time 且尚未落盘，则一次性落盘并置 save_flag。
        说明：保持开区间（start_time < t < end_time）。
        """
        if self.save_flag:
            return

        # 惰性建立本地行号映射
        if self.local_tracked_ids.size == 0:
            self.determine_local_ids(bunch_obj)
            if self.local_tracked_ids.size == 0:
                return

        # 超出结束时间 → 一次性落盘
        if not (self.start_time < time_Pre < self.end_time):
            if time_Pre > self.end_time and not self.save_flag:
                self.dump_all_particles()
                self.save_flag = True
            return

        # 仅当 time_Post 也在开区间内才继续
        if self.start_time < time_Post < self.end_time:
            # 用 floor 计算栅格索引；加一个微小 eps 抑制浮点边界抖动
            eps = 1e-12
            pre_idx  = int(np.floor((time_Pre  - self.start_time) / self.time_interval + eps))
            post_idx = int(np.floor((time_Post - self.start_time) / self.time_interval + eps))

            if pre_idx == post_idx:
                # 未跨越栅格 -> 不记录
                return
            else:
                # 跨越了至少一个栅格 -> 记录本步状态（不插值），时间记 time_Post
                self.save_all_particles(bunch_obj, data_type, time_Post)
        else:
            return

    def dump_all_particles(self):
        """
        将缓冲写入独立文件（每个粒子一个 .npz；压缩保存）。
        文件名：particle_{gid}_rank_{rank}.npz
        数组：particles -> shape (N, D+1)，最后一列为 time(秒)
        """
        total = 0
        for gid, rows in self.particle_buffer_dict.items():
            if not rows:
                print(f"[Rank {self.rank}] 粒子 {gid} 无数据，跳过。")
                continue
            data = np.vstack(rows)
            file_name = f"particle_{gid}_rank_{self.rank}.npz"
            file_path = os.path.join(self.save_folder, file_name)
            np.savez_compressed(file_path, particles=data)
            total += data.shape[0]
            print(f"[Rank {self.rank}] 粒子 {gid} 写入 {data.shape[0]} 条 -> {file_path}")
        if total == 0:
            print(f"[Rank {self.rank}] 无任何数据写入。")


class StepDumpBunch:
    def __init__(self, start_time, end_time, time_interval, filter_survive_flag = True, save_folder="dump_data"):
        """
         StepDumpBunch类，用于在 RK 积分过程中每隔一定的 step 进行一次束团 Dump操作。
        :param step_interval: int, 每隔多少步进行一次 Dump。
        :param save_folder: str, 保存文件的文件夹路径，不包含 rank，rank 号会自动添加为子文件夹。
        """
        self.start_time = start_time
        self.end_time = end_time
        self.time_interval = time_interval
        self.save_folder = save_folder
        self.filter_survive_flag = filter_survive_flag
        # 获取 MPI 线程号
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        # 创建保存路径，如果文件夹不存在则创建, 若文件夹已存在且不为空，则终止程序且给出提示
        if rank == 0:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
                print(f"文件夹 {save_folder} 已成功创建。")
            else:
                if os.listdir(save_folder):  # 检查文件夹是否为空
                    sys.stderr.write(f"Error: 数据文件夹 {save_folder} 已存在且不为空，所有进程将被终止。\n")
                    comm.Abort(1)  # 终止所有进程
                else:
                    print(f"文件夹 {save_folder} 已成功创建。")
            time.sleep(0.5)

        # 确保所有进程都同步等 rank 0 完成检查
        comm.Barrier()
        # print(f"Rank {rank} 完成文件夹检查, current time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    def check_and_dump(self, time_Pre, time_Post, bunch_obj, data_type):
        """
        检查当前时刻是否满足 Dump 条件，如果满足则进行 Dump 操作。
        :param time_Pre: 前一步时间。
        :param time_Post: 当前时间。
        :param bunch_obj: FFAG_Bunch 对象，表示当前的粒子束。
        :param data_type: 数据类型。
        """
        # 仅当 time_Post 处于可 Dump 的区间内才进行后续判断
        if self.start_time < time_Post < self.end_time:
            # 计算 time_Pre, time_Post 分别对应的索引
            time_Pre_idx = (time_Pre - self.start_time) // self.time_interval
            time_Post_idx = (time_Post - self.start_time) // self.time_interval

            if time_Pre_idx == time_Post_idx:
                # 没有跨过新的 Dump 时刻，直接返回
                return
            else:
                # 说明 (time_Pre, time_Post) 区间至少跨过一个 Dump 时刻
                self.dump_bunch(bunch_obj, time_Post, data_type)
        else:
            return

    def dump_bunch(self, bunch_obj, time_Post, data_type):
        """
        执行粒子的 Dump 操作，保存当前的粒子束信息。
        :param bunch_obj: FFAG_Bunch 对象。
        :param step_i: int, 当前步数，用于标记当前的粒子状态。
        """
        # 获取当前 MPI 线程号
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        # 获取粒子数据
        LocalBunch = bunch_obj.LocalBunch

        # 筛选存活的粒子
        survive_flag_idx = bunch_obj.BunchAttribute.Attribute['Survive']
        survive_flag = LocalBunch[:, survive_flag_idx]
        if self.filter_survive_flag:
            survived_particles = LocalBunch[survive_flag== 1, :]
        else:
            survived_particles = LocalBunch
        # 如果没有存活的粒子，则跳过
        if np.size(survived_particles, 0) == 0:
            return

        # 创建文件名注意时间转换为ns, 最好保留2位小数
        filename = f"time_{time_Post*1e9:.2f}_rank_{rank}.npz"
        filepath = os.path.join(self.save_folder, filename)

        if data_type == "RK4":
            particles_to_dump = FFAG_ConversionTools().ConvertVrzek2Przek(survived_particles)
        elif data_type == "Boris":
            particles_to_dump = FFAG_ConversionTools().ConvertVrzek2Przek_boris(survived_particles)
        else:
            raise ValueError("data_type 的值必须是 RK 或 Boris")

        # 保存存活粒子数据到文件，使用 npz 格式
        particles_with_time = np.hstack((particles_to_dump, np.full((particles_to_dump.shape[0], 1), time_Post)))
        append_to_npz(filepath, particles_with_time)


# class PositionDump:
#     def __init__(self, target_azimuth_angle_deg, start_time, end_time,
#                  tracked_particle_global_ID, save_folder="dump_data", max_records=10000):
#         """
#         PositionDump：记录指定粒子在穿越目标方位角时的状态（不插值），写入固定预分配数组。
#         :param target_azimuth_angle_deg: float，目标方位角 [0, 360]，单位为度。
#         :param start_time: float，记录开始时间。
#         :param end_time: float，记录结束时间。
#         :param tracked_particle_global_ID: 追踪粒子的GIDs, ndarray
#         :param save_folder: str，保存文件夹。
#         :param max_records: int，最多记录次数（默认10000）。
#         """
#         self.target_azimuth_angle = np.deg2rad(target_azimuth_angle_deg) % (2 * np.pi)
#         self.start_time = start_time
#         self.end_time = end_time
#         self.tracked_particle_global_ID = tracked_particle_global_ID
#         self.save_folder = save_folder
#         self.max_records = max_records
#         self.write_cursor = 0
#         self.save_flag = False
#
#         # MPI
#         comm = MPI.COMM_WORLD
#         self.rank = comm.Get_rank()
#
#         # 创建保存路径
#         if self.rank == 0:
#             if not os.path.exists(save_folder):
#                 os.makedirs(save_folder)
#                 print(f"文件夹 {save_folder} 已成功创建。")
#             elif os.listdir(save_folder):
#                 sys.stderr.write(f"Error: 数据文件夹 {save_folder} 已存在且不为空，所有进程将被终止。\n")
#                 comm.Abort(1)
#         comm.Barrier()
#
#         if not os.path.exists(self.save_folder):
#             os.makedirs(self.save_folder)
#
#         # 初始化 buffer（等粒子维度信息确定后再分配）
#         self.particle_buffer = None
#
#     @profile
#     def check_azimuth_crossing(self, bunch_obj):
#         """
#         角度穿越检测
#         """
#         pre_fi_raw = bunch_obj.PreStep_Fi
#         post_fi_raw = bunch_obj.PostStep_Fi
#         survive_flag = bunch_obj.Step_SurviveFlag
#         pre_fi_mod_malloc = bunch_obj.PreStep_FiModMalloc
#         post_fi_mod_malloc = bunch_obj.PostStep_FiModMalloc
#         crossing_bool = bunch_obj.CrossingCheckBoolMalloc
#         crossing_ratio = bunch_obj.CrossingCheckRatioMalloc
#
#         detect_azimuth_crossing(pre_fi_raw, post_fi_raw,
#                                 pre_fi_mod_malloc, post_fi_mod_malloc,
#                                 crossing_bool, crossing_ratio,
#                                 survive_flag, self.target_azimuth_angle)
#
#         return crossing_bool
#
#     @profile
#     def check_and_dump(self, step_i, time_i, bunch_obj, data_type):
#         """
#         每个时间步调用一次，判断是否记录当前粒子状态。
#         """
#         if self.save_flag:
#             return
#
#         if time_i < self.end_time and time_i > self.start_time:
#             crossing_bool = self.check_azimuth_crossing(bunch_obj) # 判断当前进程所有粒子的crossing_bool
#
#         elif time_i > self.end_time and self.save_flag is False:
#             self.save_particle()
#             self.save_flag = True
#             return
#         else:
#             return
#
#         if self.particle_buffer is None:
#             dim = bunch_obj.LocalBunch.shape[1] + 1  # +1 for time
#             self.particle_buffer = np.empty((self.max_records, dim), dtype=np.float64)
#
#         if not crossing_bool[self.tracked_pid]:
#             return
#
#         # 获取当前粒子的状态并转换
#         state = bunch_obj.LocalBunch[self.tracked_pid:self.tracked_pid + 1, :]  # shape (1, dim)
#         if data_type == "RK4":
#             converted = FFAG_ConversionTools().ConvertVrzek2Przek(state)
#         elif data_type == "Boris":
#             converted = FFAG_ConversionTools().ConvertVrzek2Przek_boris(state)
#         else:
#             raise ValueError("data_type 必须是 'RK4' 或 'Boris'")
#
#         converted[:, 9] = np.mod(converted[:, 9], 2 * np.pi)
#         full_data = np.hstack((converted, [[time_i]]))  # shape: (1, dim+1)
#
#         if self.write_cursor < self.max_records:
#             self.particle_buffer[self.write_cursor, :] = full_data[0]
#             self.write_cursor += 1
#         else:
#             print(f"[Rank {self.rank}] 已达到最大记录数量 ({self.max_records})，后续不再记录。")
#
#     def save_particle(self):
#         """
#         将记录的数据写入文件。
#         """
#         if self.write_cursor == 0:
#             print(f"[Rank {self.rank}] 粒子 {self.tracked_pid} 无穿越记录，未保存。")
#             return
#
#         data = self.particle_buffer[:self.write_cursor, :]
#         globalIDs = data[0, 15]
#         file_name = f"particle_{globalIDs}_angle_{np.rad2deg(self.target_azimuth_angle):.1f}_rank_{self.rank}.npz"
#         file_path = os.path.join(self.save_folder, file_name)
#         np.savez(file_path, particles=data)
#         print(f"[Rank {self.rank}] 粒子 {self.tracked_pid} 保存完成，共 {self.write_cursor} 条记录。")
#
#     def determine_local_ids(self, bunch_obj):
#         """
#         基于 LocalBunch 的最后一列(Global IDs)，找出当前进程需要记录的本地行号列表(Local IDs)。
#         """
#         # 1) 取出本地全局ID列（最后一列）
#         global_ids_local = bunch_obj.LocalBunch[:, -1].astype(np.int64, copy=False)
#
#         # 2) 构建 gid -> 行号 的映射
#         gid_to_row = {int(gid): row for row, gid in enumerate(global_ids_local)}
#
#         # 3) 计算本地匹配到的行号
#         self.local_tracked_ids = [gid_to_row[g] for g in self.tracked_particle_global_ID if g in gid_to_row]
#
#         # print(f"[Rank {self.rank}] 目标全局ID: {self.tracked_global_ids} -> 本地匹配行号: {self.local_tracked_ids}")
#         return self.local_tracked_ids


class PositionDump:
    def __init__(self, target_azimuth_angle_deg, start_time, end_time,
                 tracked_particle_id_global: list[int],
                 save_folder="dump_data", max_records=10000):
        """
        使用缓存记录多个粒子穿越目标方位角时的状态（不插值），每个粒子保存为独立文件。

        :param target_azimuth_angle_deg: float，目标方位角 [0, 360]，单位为度。
        :param start_time: float，记录开始时间。
        :param end_time: float，记录结束时间。
        :param tracked_particle_id_global: np.ndarray，要记录的全局粒子 ID 列表。
        :param save_folder: str，保存文件夹。
        :param max_records: int，每个粒子最多记录次数。
        """
        self.target_azimuth_angle = np.deg2rad(target_azimuth_angle_deg) % (2 * np.pi)
        self.start_time = start_time
        self.end_time = end_time
        self.tracked_global_ids = np.asarray(tracked_particle_id_global, dtype=np.int64)
        assert self.tracked_global_ids.ndim == 1, "tracked_particle_id_global 必须是 1D ndarray"

        self.save_folder = save_folder
        self.max_records = max_records
        self.save_flag = False

        # MPI
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()

        # 创建保存路径
        if self.rank == 0:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
                print(f"文件夹 {save_folder} 已创建。")
            elif os.listdir(save_folder):
                sys.stderr.write(f"Error: 数据文件夹 {save_folder} 不为空，终止所有进程。\n")
                comm.Abort(1)
        comm.Barrier()

        # 缓冲区
        self.particle_buffer_dict = {}   # gid → ndarray(max_records, dim)
        self.cursor_dict = {}            # gid → 当前写入位置
        self.write_done_dict = {}        # gid → True if 满了
        self.local_tracked_ids = np.zeros((0,))      # 本地需要记录的粒子行号

        self._base_dim = None
        self._extra_cols = None
        self._conv = FFAG_ConversionTools()

    def determine_local_ids(self, bunch_obj):
        """
        构建全局 ID 到本地行号的映射，找出本进程需记录的粒子。
        """
        global_ids_local = bunch_obj.LocalBunch[:, -1].astype(np.int64)
        gid_to_row = {gid: row for row, gid in enumerate(global_ids_local)}
        self.local_tracked_ids = [gid_to_row[gid] for gid in self.tracked_global_ids if gid in gid_to_row]
        # print(f"[Rank {self.rank}] 本地匹配行号: {self.local_tracked_ids}")
        self.local_tracked_ids=np.array(self.local_tracked_ids)
        return self.local_tracked_ids

    def _init_gid_buffer_if_needed(self, gid, dim):
        if gid not in self.particle_buffer_dict:
            self.particle_buffer_dict[gid] = np.empty((self.max_records, dim), dtype=np.float64)
            self.cursor_dict[gid] = 0
            self.write_done_dict[gid] = False

    def _append_to_gid_buffer(self, gid, data_row):
        if self.write_done_dict.get(gid, False):
            return  # 已满
        cursor = self.cursor_dict[gid]
        if cursor < self.max_records:
            self.particle_buffer_dict[gid][cursor, :] = data_row
            self.cursor_dict[gid] += 1
        else:
            print(f"[Rank {self.rank}] 粒子 {gid} 的缓存已满（{self.max_records} 条），停止记录。")
            self.write_done_dict[gid] = True

    def check_azimuth_crossing(self, bunch_obj):
        """
        检测本步粒子是否穿越目标方位角。
        """
        detect_azimuth_crossing_portion(
            bunch_obj.PreStep_Fi,
            bunch_obj.PostStep_Fi,
            bunch_obj.PreStep_FiModMalloc,
            bunch_obj.PostStep_FiModMalloc,
            bunch_obj.CrossingCheckBoolMalloc,
            bunch_obj.CrossingCheckRatioMalloc,
            bunch_obj.Step_SurviveFlag,
            self.target_azimuth_angle,
            self.local_tracked_ids
        )
        return bunch_obj.CrossingCheckBoolMalloc

    # # @profile
    # def check_and_dump(self, step_i, time_i, bunch_obj, data_type):
    #     """
    #     每个时间步调用一次，记录所有发生穿越的目标粒子状态。
    #     """
    #     if self.save_flag:
    #         return
    #
    #     if not (self.start_time < time_i < self.end_time):
    #         if time_i > self.end_time and not self.save_flag:
    #             self.save_all_particles()
    #             self.save_flag = True
    #         return
    #
    #     if np.size(self.local_tracked_ids,0)<1:
    #         self.determine_local_ids(bunch_obj)
    #         if np.size(self.local_tracked_ids,0)<1:
    #             return
    #
    #     crossing_bool = self.check_azimuth_crossing(bunch_obj)
    #
    #     base_dim = bunch_obj.LocalBunch.shape[1]
    #     if self._base_dim is None:
    #         self._base_dim = base_dim
    #         self._extra_cols = 1  # time + global_id
    #
    #     for local_id in self.local_tracked_ids:
    #         # 本地需要记录的粒子行号
    #         if not crossing_bool[local_id]:
    #             continue
    #
    #         state = bunch_obj.LocalBunch[local_id:local_id + 1, :]
    #         if data_type == "RK4":
    #             converted = self._conv.ConvertVrzek2Przek(state)
    #         elif data_type == "Boris":
    #             converted = self._conv.ConvertVrzek2Przek_boris(state)
    #         else:
    #             raise ValueError("data_type 必须为 'RK4' 或 'Boris'")
    #
    #         converted[:, 9] = np.mod(converted[:, 9], 2 * np.pi)
    #         time_arr = np.array([[time_i]])
    #         gid_arr = bunch_obj.LocalBunch[local_id:local_id + 1, -1].reshape(1, 1)
    #
    #         full_data = np.hstack((converted, time_arr))  # shape: (1, dim)
    #         gid = int(gid_arr[0, 0])
    #         self._init_gid_buffer_if_needed(gid, full_data.shape[1])
    #         self._append_to_gid_buffer(gid, full_data[0])


    # @profile
    def check_and_dump(self, step_i, time_i, bunch_obj, data_type):
        """
        每个时间步调用一次，记录所有发生穿越的目标粒子状态。
        """
        if self.save_flag:
            return

        if not (self.start_time < time_i < self.end_time):
            if time_i > self.end_time and not self.save_flag:
                self.save_all_particles()
                self.save_flag = True
            return

        if np.size(self.local_tracked_ids,0)<1:
            self.determine_local_ids(bunch_obj)
            if np.size(self.local_tracked_ids,0)<1:
                return

        crossing_bool = self.check_azimuth_crossing(bunch_obj)

        base_dim = bunch_obj.LocalBunch.shape[1]
        if self._base_dim is None:
            self._base_dim = base_dim
            self._extra_cols = 1  # time + global_id

        # —— 取出本步命中的本地行号——
        hits_mask = crossing_bool[self.local_tracked_ids]
        if not np.any(hits_mask):
            return

        hits = self.local_tracked_ids[hits_mask]  # shape: (H,)

        # —— 批量拿出状态，一次性做转换 ——
        states = bunch_obj.LocalBunch[hits, :]  # (H, base_dim)
        if data_type == "RK4":
            converted = self._conv.ConvertVrzek2Przek(states)  # (H, conv_dim)
        elif data_type == "Boris":
            converted = self._conv.ConvertVrzek2Przek_boris(states)  # (H, conv_dim)
        else:
            raise ValueError("data_type 必须为 'RK4' 或 'Boris'")

        converted[:, 9] = np.mod(converted[:, 9], 2 * np.pi)

        # 一次性拼接时间列（避免每次小的 np.array 分配）
        time_col = np.full((converted.shape[0], 1), time_i, dtype=np.float64)
        full = np.concatenate((converted, time_col), axis=1)  # (H, dim_out)

        # —— 按 gid 分组，一次性写入各自 buffer（减少 dict 反复查找/写一行一行）——
        gids = states[:, -1].astype(np.int64)  # 从 LocalBunch 末列取 gid
        order = np.argsort(gids)
        gids_sorted = gids[order]
        full_sorted = full[order]

        # 找到每个 gid 的分组边界
        starts = np.r_[0, np.flatnonzero(gids_sorted[1:] != gids_sorted[:-1]) + 1]
        ends = np.r_[starts[1:], gids_sorted.size]

        # gids_sorted[1:] != gids_sorted[:-1], 比较相邻元素是否不同, 得到一个布尔数组，例如
        # gids_sorted = [10, 10, 11, 11, 11, 12]
        # gids_sorted[1:] != gids_sorted[:-1]
        # -> [False, True, False, False, True]

        # np.flatnonzero(...)返回非零（True）的索引：[1, 4]（也就是变化发生的位置）
        # +1因为比较的是1: 和:-1，变化实际发生在后一项位置上。
        # np.r_[0, ...]在最前面加上0（第一个分组总是从0开始）

        # starts = [0, 2, 5]
        # ends = [2, 5, 6]


        # 按粒子gid为单位，把属于这个粒子的若干行（block）从批量结果full_sorted
        # 中写入该粒子的缓存数组中（particle_buffer_dict[gid]），同时维护当前写入位置
        # cursor_dict[gid]和 是否写满的标志write_done_dict[gid]
        for s, e in zip(starts, ends):
            gid = int(gids_sorted[s])
            block = full_sorted[s:e, :]  # (k_i, dim_out)
            self._init_gid_buffer_if_needed(gid, block.shape[1])
            # 现在每个粒子都有自己的表格，用来存放它多次穿越记录
            cursor = self.cursor_dict[gid]
            n_room = self.max_records - cursor
            # 当前这个粒子已经写入了多少行（写指针位置）,计算还剩多少可写空间
            # 如果满了，跳过这个粒子（不再记录）
            if n_room <= 0:
                self.write_done_dict[gid] = True
                continue
            n = block.shape[0] if block.shape[0] <= n_room else n_room
            # 计算本次要写的行数, block.shape[0] 是这一轮命中的条数。
            # 如果命中行比剩余空间还多，就截断到可写空间
            self.particle_buffer_dict[gid][cursor:cursor + n, :] = block[:n, :]
            self.cursor_dict[gid] = cursor + n

            if self.cursor_dict[gid] >= self.max_records:
                self.write_done_dict[gid] = True


    def save_all_particles(self):
        """
        将每个粒子的缓存写入独立文件。
        """
        for gid, buffer in self.particle_buffer_dict.items():
            count = self.cursor_dict.get(gid, 0)
            if count == 0:
                print(f"[Rank {self.rank}] 粒子 {gid} 无数据，跳过保存。")
                continue
            data = buffer[:count, :]
            angle_deg = np.rad2deg(self.target_azimuth_angle)
            file_name = f"particle_{gid}_angle_{angle_deg:.1f}_rank_{self.rank}.npz"
            file_path = os.path.join(self.save_folder, file_name)
            np.savez(file_path, particles=data)
            print(f"[Rank {self.rank}] 粒子 {gid} 已保存 {count} 条记录。")


class PositionDumpBunch:
    def __init__(self, target_azimuth_angle_deg, start_time, end_time, save_folder="dump_data"):
        """
        PositionDump类，用于粒子穿越给定方位角时进行 Dump 操作, 按粒子进行区分，不同粒子保存为一个单独文件。
        :param target_azimuth_angle_deg: float, Dump 的方位角 [0, 360]。
        :param num_particles_to_dump_global: int, 每次 Dump 保存的粒子总数（全局范围）。
        :param save_folder: str, 保存文件的文件夹路径，不包含 rank，rank 号会自动添加为子文件夹。
        """
        self.target_azimuth_angle = np.deg2rad(target_azimuth_angle_deg) % (2 * np.pi)  # 转换为弧度并归一化到 [0, 2π)
        self.save_folder = save_folder
        self.start_time = start_time
        self.end_time = end_time
        # self.saved_data = np.zeros((0, 17))  # 存储所有待保存的粒子数据
        self.save_flag = False

        self.saved_data = np.empty((50000, 17), dtype=np.float64)  # 初始容量
        self.write_cursor = 0  # 当前写入位置

        # 获取 MPI 通信器
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        # 创建保存路径，按 rank 创建子文件夹
        # self.local_folder_path = os.path.join(save_folder, f"rank_{self.rank}")
        if self.rank == 0:
            # print(f"文件夹 {save_folder} 已成功创建。")
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
                print(f"文件夹 {save_folder} 已成功创建。")
            else:
                if os.listdir(save_folder):  # 检查文件夹是否为空
                    sys.stderr.write(f"Error: 数据文件夹 {save_folder} 已存在且不为空，所有进程将被终止。\n")
                    comm.Abort(1)
        # print(f"[rank {self.rank}] init dump_folder={save_folder}", flush=True)
        # print(f"[rank {self.rank}] BEFORE barrier dump_folder={save_folder}", flush=True)
        # comm.Barrier()
        # print(f"[rank {self.rank}] AFTER barrier dump_folder={save_folder}", flush=True)


        # if not os.path.exists(self.save_folder):
            # os.makedirs(self.save_folder)
        # if self.rank == 0:
        #     print(f"Rank {self.rank} 完成文件夹检查, 当前时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    # @profile
    # def check_azimuth_crossing(self, bunch_obj):
    #     """
    #     检查粒子是否穿越了指定的目标方位角 (self.target_azimuth_angle)。
    #     """
    #
    #     # 1) 直接从 bunch_obj 引用
    #     pre_fi_raw = bunch_obj.Pre_steps[:, 3]
    #     post_fi_raw = bunch_obj.Post_steps[:, 3]
    #     survive_flag = bunch_obj.Post_steps[:, 4]
    #
    #     target_angle = self.target_azimuth_angle % (2 * np.pi)
    #
    #     # 2) 就地将角度归一化到 [0, 2π)，无需 copy
    #     pre_fi_raw %= (2 * np.pi)  # 修改原数组
    #     post_fi_raw %= (2 * np.pi)  # 修改原数组
    #
    #     # 对于后续计算，为了让变量名更明确，可以直接重命名引用：
    #     pre_fi = pre_fi_raw
    #     post_fi = post_fi_raw
    #
    #     # 3) 准备输出 ratio_merge
    #     ratio_merge = np.full(pre_fi.shape, np.nan, dtype=np.float64)
    #
    #     # 4) 针对存活粒子做进一步计算
    #     alive_mask = (survive_flag == 1)
    #     if not np.any(alive_mask):
    #         return np.array([], dtype=int), ratio_merge
    #
    #     alive_idx = np.flatnonzero(alive_mask)
    #     pre_fi_alive = pre_fi[alive_idx]
    #     post_fi_alive = post_fi[alive_idx]
    #
    #     # 5) 根据是否跨过 2π 对 post_fi_eff 进行修正
    #     post_fi_eff = post_fi_alive.copy()  # 在这里仍需要一个小的 copy
    #     cross_mask = (post_fi_alive < pre_fi_alive)
    #     post_fi_eff[cross_mask] += 2 * np.pi
    #
    #     # 6) 判断是否穿越
    #     span = post_fi_eff - pre_fi_alive
    #     valid_span = (span != 0.0)
    #     crossing_mask_alive = (
    #             (pre_fi_alive <= target_angle) &
    #             (target_angle <= post_fi_eff) &
    #             valid_span
    #     )
    #
    #     ratio_alive = np.full_like(pre_fi_alive, np.nan, dtype=np.float64)
    #     ratio_alive[crossing_mask_alive] = (
    #             (target_angle - pre_fi_alive[crossing_mask_alive]) /
    #             span[crossing_mask_alive]
    #     )
    #
    #     # 7) 写回
    #     ratio_merge[alive_idx] = ratio_alive
    #     final_crossing_idx_alive = alive_idx[crossing_mask_alive]
    #     return final_crossing_idx_alive, ratio_merge

    # @profile
    def check_azimuth_crossing(self, bunch_obj):
        """
        检查粒子是否穿越了指定的目标方位角 (self.target_azimuth_angle)。
        :param bunch_obj: FFAG_Bunch 对象，包含粒子的状态信息。
        :return:
            crossing_indices : np.ndarray[int], 在本步中穿越目标方位角的粒子索引
            ratio_merge      : np.ndarray[float], 对应每个粒子的插值比例, 未穿越则 NaN
        """

        # -------------------- 1) 提取数据 --------------------
        pre_fi_raw = bunch_obj.PreStep_Fi # 上一步的方位角
        post_fi_raw = bunch_obj.PostStep_Fi  # 当前步的方位角
        survive_flag = bunch_obj.Step_SurviveFlag  # 粒子存活标志 (1 表示存活)
        pre_fi_mod_malloc = bunch_obj.PreStep_FiModMalloc
        post_fi_mod_malloc = bunch_obj.PostStep_FiModMalloc
        crossing_bool = bunch_obj.CrossingCheckBoolMalloc
        crossing_ratio = bunch_obj.CrossingCheckRatioMalloc

        # pre_fi = fast_mod_njit(pre_fi_raw, (2 * np.pi))
        # post_fi = fast_mod_njit(post_fi_raw, (2 * np.pi))
        # fast_mod_parallel(pre_fi_raw, (2 * np.pi), pre_fi_mod_malloc)
        # fast_mod_parallel(post_fi_raw, (2 * np.pi), post_fi_mod_malloc)

        detect_azimuth_crossing(pre_fi_raw, post_fi_raw,
                                pre_fi_mod_malloc, post_fi_mod_malloc,
                                crossing_bool, crossing_ratio,
                                survive_flag, self.target_azimuth_angle)

        # target_angle = self.target_azimuth_angle % (2 * np.pi)
        #
        # # -------------------- 2) 归一化方位角到 [0, 2π) --------------------
        # # pre_fi = pre_fi_raw % (2 * np.pi)
        # # post_fi = post_fi_raw % (2 * np.pi)
        #
        # # pre_fi = np.mod(pre_fi_raw, (2 * np.pi))
        # # post_fi = np.mod(post_fi_raw, (2 * np.pi))
        # pre_fi = fast_mod_njit(pre_fi_raw, (2 * np.pi))
        # post_fi = fast_mod_njit(post_fi_raw, (2 * np.pi))
        #
        # # 准备输出： ratio_merge 默认 NaN
        # ratio_merge = np.full(pre_fi.shape, np.nan, dtype=np.float64)
        #
        # # -------------------- 3) 情形 A: post_fi >= pre_fi --------------------
        # cond_A = (post_fi >= pre_fi)
        # # A1: 判断穿越 (pre_fi <= target_angle <= post_fi)
        # crossing_mask_A = (
        #         cond_A &
        #         (pre_fi <= target_angle) &
        #         (target_angle <= post_fi) &
        #         (survive_flag == 1)
        # )
        # # A2: 计算插值比例 ratio = (target_angle - pre_fi) / (post_fi - pre_fi)
        # span_A = post_fi - pre_fi
        # valid_A = (span_A != 0)
        # ratio_A = np.full_like(span_A, np.nan, dtype=np.float64)
        # mask_A_final = crossing_mask_A & valid_A
        # ratio_A[mask_A_final] = (target_angle - pre_fi[mask_A_final]) / span_A[mask_A_final]
        #
        # # -------------------- 4) 情形 B: post_fi < pre_fi (跨 2π) --------------------
        # cond_B = (post_fi < pre_fi)
        # # B1: 判断穿越
        # #     若目标角 ∈ [pre_fi, 2π) 或 [0, post_fi], 则表示穿越
        # crossing_mask_B = (
        #         cond_B &
        #         (survive_flag == 1) &
        #         (
        #                 ((target_angle >= pre_fi) & (target_angle < 2 * np.pi))  # [pre_fi, 2π)
        #                 |
        #                 ((target_angle >= 0) & (target_angle <= post_fi))  # [0, post_fi]
        #         )
        # )
        # # B2: 计算插值比例
        # #     将 post_fi 视为 post_fi+2π, 若目标角 < post_fi 则也视为 target_angle+2π
        # post_fi_eff = np.where(cond_B, post_fi + 2 * np.pi, post_fi)
        # target_angle_eff = np.where(
        #     cond_B & (target_angle <= post_fi),
        #     target_angle + 2 * np.pi,
        #     target_angle
        # )
        # span_B = post_fi_eff - pre_fi  # 跨度
        # ratio_B = np.full_like(span_B, np.nan, dtype=np.float64)
        # valid_B = (span_B != 0)
        # mask_B_final = crossing_mask_B & valid_B
        # ratio_B[mask_B_final] = (target_angle_eff[mask_B_final] - pre_fi[mask_B_final]) / span_B[mask_B_final]
        #
        # # -------------------- 5) 合并两次判断 --------------------
        # final_crossing_mask = crossing_mask_A | crossing_mask_B
        # crossing_indices = np.where(final_crossing_mask)[0]
        #
        # # 把 ratio_A, ratio_B 分别写入 ratio_merge
        # ratio_merge[crossing_mask_A] = ratio_A[crossing_mask_A]
        # ratio_merge[crossing_mask_B] = ratio_B[crossing_mask_B]
        #
        # # -------------------- 6) 返回结果 --------------------
        return crossing_bool


    def add_particles(self, particle_data, time_i):
        """
        保存单个粒子的状态数据到文件，并在最后一列添加时间信息。
        :param particle_data: ndarray, 粒子的状态数据。
        :param time_i: float, 当前时间，用于标记粒子状态。
        """
        #
        # # 在粒子数据的最后一列附加时间信息
        # # particle_with_time = np.append(particle_data, time_i)
        # particle_with_time = np.column_stack((particle_data, np.full((particle_data.shape[0],), time_i)))
        # # 将数据添加到矩阵中
        # self.saved_data = np.row_stack((self.saved_data, particle_with_time))

        N_new = particle_data.shape[0]
        time_col = np.full((N_new, 1), time_i, dtype=np.float64)
        particle_with_time = np.hstack((particle_data, time_col))  # shape: (N_new, 17)

        # 如果空间不够，扩容（翻倍策略）
        required = self.write_cursor + N_new
        # print("required=", required)
        if required > self.saved_data.shape[0]:
            new_capacity = max(required, self.saved_data.shape[0] * 2)
            new_buffer = np.empty((new_capacity, 17), dtype=np.float64)
            new_buffer[:self.write_cursor, :] = self.saved_data[:self.write_cursor, :]
            self.saved_data = new_buffer

        # 填入数据
        self.saved_data[self.write_cursor:self.write_cursor + N_new, :] = particle_with_time
        self.write_cursor += N_new


    def save_all_particles(self, data_type):
        """
        一次性保存所有粒子数据到文件。
        """
        # 如果没有数据，直接返回
        if len(self.saved_data) == 0:
            return

        # 合并所有粒子数据为一个大矩阵
        # all_particle_data = np.array(self.saved_data)
        all_particle_data = self.saved_data[:self.write_cursor, :]

        # 创建文件名，包含方位角、start_time、end_time和rank
        file_name = f"crossing_angle_{np.rad2deg(self.target_azimuth_angle):.1f}_start_{self.start_time*1e9:.3f}_end_{self.end_time*1e9:.3f}_rank_{self.rank}.npz"
        file_path = os.path.join(self.save_folder, file_name)

        # 根据选定的索引保存对应粒子的信息
        if data_type == "RK4":
            particles_to_dump = FFAG_ConversionTools().ConvertVrzek2Przek(all_particle_data[:,:-1])
        elif data_type == "Boris":
            particles_to_dump = FFAG_ConversionTools().ConvertVrzek2Przek_boris(all_particle_data[:,:-1])
        else:
            raise ValueError("data_type 的值必须是 RK 或 Boris")

        particles_to_dump[:,9] = np.mod(particles_to_dump[:,9], 2*np.pi)
        particles_to_dump = np.hstack((particles_to_dump, all_particle_data[:,[-1]]))

        # 保存数据到文件
        np.savez(file_path, particles = particles_to_dump)

    # @profile
    # def check_and_dump(self, step_i, time_i, bunch_obj, data_type):
    #     """
    #     检查粒子是否穿越目标方位角，并保存符合条件的粒子数据。
    #     :param step_i: int, 当前步数。
    #     :param time_i: float, 当前时间。
    #     :param bunch_obj: FFAG_Bunch 对象，表示当前的粒子束。
    #     """
    #
    #     # 获取穿越目标方位角的粒子索引
    #     if time_i < self.end_time and time_i > self.start_time:
    #         crossing_indices, crossing_ratio = self.check_azimuth_crossing(bunch_obj)
    #     elif time_i > self.end_time and self.save_flag is False:
    #         self.save_all_particles()
    #         self.save_flag = True
    #         return
    #     else:
    #         return
    #
    #     # 如果没有粒子穿越，直接返回
    #     if len(crossing_indices) == 0:
    #         return
    #
    #     # 从粒子束中筛选这些粒子的状态数据
    #     selected_particles_PostStep = bunch_obj.LocalBunch[crossing_indices, :]
    #     selected_particles_PreStep = bunch_obj.LocalBunchPreStep[crossing_indices, :]
    #     selected_particles = selected_particles_PreStep.copy()  # 先整体复制为 PreStep
    #
    #     # 对前7列做线性插值：
    #     selected_particles[:, :7] = (
    #             selected_particles_PreStep[:, :7]
    #             + np.tile(crossing_ratio[crossing_indices].reshape(-1, 1), (1, 7))
    #             * (selected_particles_PostStep[:, :7] - selected_particles_PreStep[:, :7])
    #     )
    #
    #     # 根据选定的索引保存对应粒子的信息
    #     if data_type == "RK4":
    #         particles_to_dump = FFAG_ConversionTools().ConvertVrzek2Przek(selected_particles)
    #     elif data_type == "Boris":
    #         particles_to_dump = FFAG_ConversionTools().ConvertVrzek2Przek_boris(selected_particles)
    #     else:
    #         raise ValueError("data_type 的值必须是 RK 或 Boris")
    #     # selected_particles = FFAG_ConversionTools().ConvertVrzek2Przek(selected_particles)
    #
    #     self.add_particles(particles_to_dump, time_i)

    # @profile
    def check_and_dump(self, step_i, time_i, bunch_obj, data_type):
        """
        检查粒子是否穿越目标方位角，并保存符合条件的粒子数据。
        :param step_i: int, 当前步数。
        :param time_i: float, 当前时间。
        :param bunch_obj: FFAG_Bunch 对象，表示当前的粒子束。
        """

        # 获取穿越目标方位角的粒子索引
        if time_i < self.end_time and time_i > self.start_time:
            crossing_indices = self.check_azimuth_crossing(bunch_obj)
        elif time_i > self.end_time and self.save_flag is False:
            self.save_all_particles(data_type)
            self.save_flag = True
            return
        else:
            return

        # 如果没有粒子穿越，直接返回
        if len(crossing_indices) == 0:
            return

        # 从粒子束中筛选这些粒子的状态数据
        selected_particles = bunch_obj.LocalBunch[crossing_indices, :]

        # # 根据选定的索引保存对应粒子的信息
        # if data_type == "RK4":
        #     particles_to_dump = FFAG_ConversionTools().ConvertVrzek2Przek(selected_particles)
        # elif data_type == "Boris":
        #     particles_to_dump = FFAG_ConversionTools().ConvertVrzek2Przek_boris(selected_particles)
        # else:
        #     raise ValueError("data_type 的值必须是 RK 或 Boris")

        self.add_particles(selected_particles, time_i)


# class PositionDumpBunch:
#     def __init__(self, target_azimuth_angle_deg, save_folder="dump_data", dump_turns=None):
#         """
#         PositionDump类，用于粒子穿越给定方位角时进行 Dump 操作。
#         :param target_azimuth_angle: float, 目标方位角 (rad)。
#         :param dump_turns: list or np.ndarray, 指定需要进行 Dump 操作的圈数。
#         """
#         self.target_azimuth_angle = np.deg2rad(target_azimuth_angle_deg) % (2 * np.pi)  # 将方位角归一化到 [0, 2π) 范围内
#         self.dump_turns = dump_turns  # 用户可以提供任意圈数的列表
#         self.folder_path = save_folder
#
#         # 获取 MPI 线程号
#         comm = MPI.COMM_WORLD
#         rank = comm.Get_rank()
#         # 创建保存路径，如果文件夹不存在则创建, 若文件夹已存在且不为空，则终止程序且给出提示
#         if rank == 0:
#             if not os.path.exists(save_folder):
#                 os.makedirs(save_folder)
#                 print(f"文件夹 {save_folder} 已成功创建。")
#             else:
#                 if os.listdir(save_folder):  # 检查文件夹是否为空
#                     sys.stderr.write(f"Error: 数据文件夹 {save_folder} 已存在且不为空，所有进程将被终止。\n")
#                     comm.Abort(1)  # 终止所有进程
#                 else:
#                     print(
#                         f"文件夹 {save_folder} 已存在但为空，可以继续使用。current time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
#             time.sleep(0.5)
#
#         # 确保所有进程都同步等 rank 0 完成检查
#         comm.Barrier()
#         print(f"Rank {rank} 完成文件夹检查, current time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
#
#     def check_azimuth_crossing(self, bunch_obj):
#         """
#         检查粒子是否穿越了指定的目标方位角，并且检查是否满足圈数筛选条件
#         每个线程独立运行，独立检查每个线程的local bunch
#         返回一个1维ndarray，满足条件的粒子返回圈数，不满足条件的粒子返回-1
#         使用ndarray向量化运算，避免使用for循环
#         """
#         pre_fi = bunch_obj.Pre_steps[:, 3]  # Pre-step的方位角
#         post_fi = bunch_obj.Post_steps[:, 3]  # Post-step的方位角
#         survive_flag = bunch_obj.Post_steps[:, 4]  # Post-step的存活标志
#
#         # 获取粒子的方位角
#         pre_fi_valid = pre_fi % (2 * np.pi)  # 将方位角归一化到 [0, 2π) 范围内
#         post_fi_valid = post_fi % (2 * np.pi)
#
#         # 获取有效粒子的圈数
#         post_turn_valid = post_fi // (2 * np.pi)
#
#         # 初始化结果数组，默认所有粒子不满足条件，即返回值为 -1
#         result_turns = np.full(len(pre_fi), -1)
#
#         # pre_fi_valid < target_fi, post_fi_valid >= target_fi 表示粒子从 Pre_steps 到 Post_steps 之间跨越了目标方位角
#         # survive_flag == 1 表示粒子存活
#         # np.isin(post_turn_valid, self.dump_turns) 表示粒子的圈数在指定的 dump_turns 中
#         crossing_mask_valid = (
#                 (pre_fi_valid < self.target_azimuth_angle) &
#                 (post_fi_valid >= self.target_azimuth_angle) &
#                 (survive_flag == 1)
#         )
#
#         # 将穿越目标方位角的粒子圈数更新到结果数组中
#         result_turns[crossing_mask_valid] = post_turn_valid[crossing_mask_valid]
#
#         return result_turns  # 返回一个包含圈数的列表，未满足条件的粒子返回 -1
#
#     def dump_crossing_particles_to_files(self, bunch_obj, result_turns, time_i):
#         """
#         将穿越目标方位角的粒子数据保存为 .npz 文件。每个方位角、圈数和 MPI 线程号对应一个文件。
#         :param bunch_obj: FFAG_Bunch 对象，包含粒子信息
#         :param result_turns: ndarray，表示粒子的穿越给定方位角信息和圈数信息
#         """
#         # 获取当前 MPI 线程号
#         comm = MPI.COMM_WORLD
#         rank = comm.Get_rank()
#
#         # 获取粒子信息
#         LocalBunch = bunch_obj.LocalBunch
#         # 根据选定的索引保存对应粒子的信息
#         LocalBunch = FFAG_ConversionTools().ConvertVrzek2Przek(LocalBunch)
#
#         # 筛选出所有穿越目标方位角的粒子（result_turns != -1）
#         valid_particles_mask = result_turns != -1
#         valid_particles_data = LocalBunch[valid_particles_mask, :]  # 符合条件的粒子数据
#         valid_turns = result_turns[valid_particles_mask]  # 符合条件的圈数信息
#
#         # 如果没有符合条件的粒子，则跳过 dump
#         if len(valid_turns) == 0:
#             return
#
#         # 遍历所有符合条件的粒子，按圈数分别进行 dump
#         for turn in np.unique(valid_turns):
#             # 筛选出当前圈数的粒子
#             turn_mask = valid_turns == turn
#             turn_particles_data = valid_particles_data[turn_mask, :]
#
#             # 创建文件名
#             filename = f"crossing_angle_{np.rad2deg(self.target_azimuth_angle):.1f}_turn_{int(turn)}_rank_{rank}.npz"
#             filepath = os.path.join(self.folder_path, filename)
#
#             # 将粒子数据保存为 .npz 格式
#             particles_with_time = np.hstack((turn_particles_data, np.full((turn_particles_data.shape[0], 1), time_i)))
#             append_to_npz(filepath, particles_with_time)
#
#
#     def check_and_dump(self, step_i, time_i, bunch_obj):
#         """
#         检查粒子是否穿越指定的目标方位角，如果满足条件则保存相关数据。
#
#         :param step_i: int, 当前模拟步数。
#         :param bunch_obj: FFAG_Bunch 对象，包含粒子数据。
#         """
#         # 检查粒子是否穿越目标方位角，并返回圈数信息
#         result_turns = self.check_azimuth_crossing(bunch_obj)
#
#         # 如果有粒子满足穿越条件，则进行数据保存
#         if np.any(result_turns != -1):  # 如果有满足条件的粒子
#             self.dump_crossing_particles_to_files(bunch_obj, result_turns, time_i)


class Dumps:
    def __init__(self):
        """
        Dumps 类，用于管理多个独立的 Dump 实例。
        """
        self.dumps = []  # 存储所有 dump 实例

    def add_dump(self, dump_obj):
        """
        添加一个 dump 实例。
        :param dump_obj: 一个 dump 实例，例如 StepDump, StepDumpBunch 或 PositionDump。
        """
        self.dumps.append(dump_obj)

    def check_and_dump(self, time_Pre, time_Post, bunch_obj, data_type='RK4'):
        """
        遍历所有 dump 实例，分别调用其 check_and_dump 方法进行数据存储。
        :param time_Pre: float, 前一步时间
        :param time_Post: float, 当前时间
        :param bunch_obj: FFAG_Bunch 对象，表示当前的粒子束。
        """
        for dump in self.dumps:
            dump.check_and_dump(time_Pre, time_Post, bunch_obj, data_type)

    def list_dumps(self):
        """
        列出所有已添加的 dump 实例及其类型。
        """
        for i, dump in enumerate(self.dumps):
            print(f"Dump {i + 1}: {dump.__class__.__name__}")


def process_merge_files(file_group, output_folder, bunch_attribute):
    """
    处理分组文件并合并到 output_folder
    """
    merged_data = []
    for file in file_group:
        try:
            with np.load(file) as npz_file:
                data = npz_file['particles']
                merged_data.append(data)
        except Exception as e:
            print(f"Error loading file {file}: {e}")
            continue

    if merged_data:
        merged_data = np.vstack(merged_data)
        # 获取 Global_ID 列的索引并排序（如果需要）
        global_id_index = list(bunch_attribute.Attribute.keys()).index('Global_ID')
        if len(np.unique(merged_data[:, global_id_index])) > 1:
            sorted_indices = np.argsort(merged_data[:, global_id_index])
            merged_data = merged_data[sorted_indices]

        # 保存文件
        output_file = os.path.join(output_folder, f"{os.path.basename(file_group[0])}_merged.csv")
        column_formats = [bunch_attribute.AttributeFormat[attr] for attr in bunch_attribute.Attribute.keys()] + ['%.8e']
        np.savetxt(output_file, merged_data, delimiter=',', fmt=tuple(column_formats))
        return output_file
    return None

def merge_files_in_folder_mpi(folder_path):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 读取所有文件名（仅在主进程执行）
    if rank == 0:
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npz')]
        file_groups = {}  # 分组文件字典
        for file in files:
            key = '_'.join(os.path.basename(file).split('_')[:-1])
            if key not in file_groups:
                file_groups[key] = []
            file_groups[key].append(file)

        # 将文件组分割给每个进程
        file_group_list = list(file_groups.values())
        chunk_size = len(file_group_list) // size
        chunks = [file_group_list[i * chunk_size:(i + 1) * chunk_size] for i in range(size)]
    else:
        chunks = None

    # 广播任务分配
    chunks = comm.scatter(chunks, root=0)

    # 各进程处理分配到的文件组
    output_folder = os.path.join(folder_path, "merged_files_rank_{}".format(rank))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    bunch_attribute = FFAG_ManageBunchAttribute()  # 假设所有进程共享同样的定义
    local_results = []
    for group in chunks:
        result = process_merge_files(group, output_folder, bunch_attribute)
        if result:
            local_results.append(result)

    # 收集结果
    results = comm.gather(local_results, root=0)

    if rank == 0:
        print(f"Merging completed. Results from all ranks: {results}")


def merge_files_in_folder(folder_path):
    """
    合并文件夹中的子文件，将文件名中除了 rank 不同但其他部分相同的文件合并为一个文件。
    合并后的文件保存在输入文件夹下的子文件夹 'merged_files' 中。
    """
    file_dict = {}
    merged_folder = os.path.join(folder_path, "merged_files")  # 合并后的文件存放子文件夹

    # 创建存放合并文件的子文件夹
    if not os.path.exists(merged_folder):
        os.makedirs(merged_folder)

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.npz'):
            # 移除文件名中的 rank 信息，作为 key 存入字典
            file_key = '_'.join(filename.split('_')[:-1])  # 移除 rank 部分
            if file_key not in file_dict:
                file_dict[file_key] = []
            file_dict[file_key].append(os.path.join(folder_path, filename))

    # 对于字典中的每组文件，进行合并
    for file_key, file_list in file_dict.items():
        merged_data = []
        output_file = f"{file_key}_merged.csv"  # 生成合并后的文件名
        output_filepath = os.path.join(merged_folder, output_file)

        # 遍历每个文件并加载数据
        for file in file_list:
            try:
                # 读取 npz 文件中键为 'particles' 的数据
                with np.load(file) as npz_file:
                    data = npz_file['particles']
                    merged_data.append(data)
            except Exception as e:
                print(f"Error loading file {file}: {e}")
                continue

        bunch_attribute = FFAG_ManageBunchAttribute()
        # 按 FFAG_ManageBunchAttribute 中的定义顺序和格式生成保存格式
        attribute_names = list(bunch_attribute.Attribute.keys())  # 属性名列表（按顺序）
        column_formats = [bunch_attribute.AttributeFormat[attr] for attr in attribute_names] + ['%.8e']  # 属性对应的格式列表

        # 将所有数据合并并保存为一个文件
        if merged_data:  # 确保有数据可以合并
            merged_data = np.vstack(merged_data)

            # 获取 Global_ID 列的索引
            global_id_index = list(bunch_attribute.Attribute.keys()).index('Global_ID')

            # 检查 Global_ID 列是否有多个不同的值
            if len(np.unique(merged_data[:, global_id_index])) > 1:
                # 按照 Global_ID 列排序（从小到大）
                sorted_indices = np.argsort(merged_data[:, global_id_index])  # 获取排序后的索引
                merged_data_sorted = merged_data[sorted_indices]  # 按照索引对数据进行排序
                print(f"Sorted data based on Global_ID.")
            else:
                merged_data_sorted = merged_data  # 如果 Global_ID 都相同，不进行排序
                print(f"Global_ID values are the same, skipping sorting.")

            # 使用生成的格式保存文件
            np.savetxt(output_filepath, merged_data_sorted, delimiter=',', fmt=tuple(column_formats))

            print(f"Merged {len(file_list)} files into {output_filepath}")
        else:
            print(f"No valid data found for {file_key}, skipping merge.")


def append_to_npz(filepath, new_data):
    """
    追加新数据到 .npz 文件。如果文件已存在，加载原有数据并合并后保存；否则直接保存新数据。

    :param filepath: str, .npz 文件路径。
    :param new_data: ndarray, 要追加的数据。
    """
    if os.path.exists(filepath):
        # 加载已有数据
        existing_data = np.load(filepath)
        existing_particles = existing_data['particles']

        # 合并数据
        combined_data = np.vstack((existing_particles, new_data))
    else:
        # 文件不存在时，直接保存新数据
        combined_data = new_data

    # 重新保存合并后的数据
    np.savez_compressed(filepath, particles=combined_data)
    # print(f"Data saved to {filepath}. Total particles: {combined_data.shape[0]}")


def discover_and_group_files(folder_path):
    """
    只在 rank=0 调用：
    扫描文件夹，收集所有 npz 文件，并按照去掉最后 rank 标记的方式分组。
    返回一个 dict: { file_key: [file1, file2, ...], ... }
    """
    file_dict = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.npz'):
            file_key = '_'.join(filename.split('_')[:-1])  # 去掉最后一段，例如 rank
            if file_key not in file_dict:
                file_dict[file_key] = []
            file_dict[file_key].append(os.path.join(folder_path, filename))
    return file_dict


def merge_file_group(file_key, file_list, output_folder):
    """
    将同一个 file_key 下的所有 npz 文件合并为一个 CSV 文件。
    output_folder 是输出的文件夹路径。
    """
    merged_data = []
    for fpath in file_list:
        try:
            with np.load(fpath) as npz_file:
                data = npz_file['particles']
                merged_data.append(data)
        except Exception as e:
            print(f"[Warning] Rank {MPI.COMM_WORLD.Get_rank()} cannot load {fpath}, Error: {e}")
            continue

    if len(merged_data) == 0:
        print(f"[Info] Rank {MPI.COMM_WORLD.Get_rank()} found no valid data for {file_key}")
        return

    # 合并为一个数组
    merged_data = np.vstack(merged_data)

    # 排序
    bunch_attribute = FFAG_ManageBunchAttribute()
    attribute_names = list(bunch_attribute.Attribute.keys())
    column_formats = [bunch_attribute.AttributeFormat[attr] for attr in bunch_attribute.Attribute.keys()] + ['%.8e']

    # 对 Global_ID 排序
    global_id_index = attribute_names.index('Global_ID')
    global_ids = merged_data[:, global_id_index]
    if len(np.unique(global_ids)) > 1:
        sorted_indices = np.argsort(global_ids)
        merged_data = merged_data[sorted_indices]
        print(f"[Info] Rank {MPI.COMM_WORLD.Get_rank()} sorted data for {file_key} by Global_ID.")

    # 保存为 CSV
    output_filename = f"{file_key}_merged.csv"
    output_path = os.path.join(output_folder, output_filename)
    np.savetxt(output_path, merged_data, delimiter=",", fmt=column_formats)
    print(f"[Info] Rank {MPI.COMM_WORLD.Get_rank()} merged {len(file_list)} files -> {output_path}")


def parallel_merge_files_in_folder(folder_path):
    """
    利用 MPI 将文件合并任务分发给多个进程执行。
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Rank 0 负责收集与分组
    if rank == 0:
        file_dict = discover_and_group_files(folder_path)
        # 将字典转化成 (file_key, file_list) 的列表，方便分割分发
        grouped_items = list(file_dict.items())
    else:
        grouped_items = None

    # 同步：广播 group 的总数量
    if rank == 0:
        total_groups = len(grouped_items)
    else:
        total_groups = 0
    total_groups = comm.bcast(total_groups, root=0)

    # Rank 0 把分组列表广播或 Scatter 给各个进程
    if rank == 0:
        # 按照进程数把分组均匀分给各个 rank
        chunk_size = (total_groups + size - 1) // size  # 天花板整除
        chunks = [grouped_items[i * chunk_size: (i + 1) * chunk_size] for i in range(size)]
    else:
        chunks = None

    my_chunk = comm.scatter(chunks, root=0)
    # my_chunk 就是当前进程负责的一部分 (file_key, file_list)

    # 每个进程都要在输出文件夹下写结果，这里统一用 merged_files 文件夹
    merged_folder = os.path.join(folder_path, "merged_files")
    if rank == 0:
        if not os.path.exists(merged_folder):
            os.makedirs(merged_folder)

    # 保证子文件夹创建完毕再往下执行
    comm.barrier()

    # 每个进程处理分配到的那些分组
    for (file_key, file_list) in my_chunk:
        merge_file_group(file_key, file_list, merged_folder)

    comm.barrier()
    if rank == 0:
        print("[Info] All ranks have finished merging.")


if __name__ == "__main__":
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Merge files in a folder.")
    parser.add_argument("folder_path", type=str, help="The path to the folder containing files to merge.")
    args = parser.parse_args()

    # 调用合并函数
    merge_files_in_folder(args.folder_path)

# # 测试程序
# if __name__ == "__main__":
#     # 初始化 MPI
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#
#     # 定义一个模拟的粒子分布
#     np.random.seed(int(MPI.Wtime() * 1000) + rank)  # 使用当前时间和rank作为随机种子
#     ParticleNum = 1000
#     ParticlesDistribution = np.column_stack((
#         np.random.random(ParticleNum) * 10,  # r
#         np.random.random(ParticleNum),  # vr
#         np.random.random(ParticleNum) * 10,  # z
#         np.random.random(ParticleNum),  # vz
#         np.random.random(ParticleNum) * 2 * np.pi,  # fi (方位角)
#         np.random.random(ParticleNum) * 100,  # Ek
#         np.random.random(ParticleNum) * 5,  # inj_t
#         np.ones(ParticleNum),  # survive flag
#         np.random.random(ParticleNum) * 2 * np.pi,  # RF_phase
#         np.zeros(ParticleNum),  # Esc_r (initially zero)
#         np.zeros(ParticleNum),  # Esc_z (initially zero)
#         np.zeros(ParticleNum),  # Esc_fi (initially zero)
#         np.zeros(ParticleNum),  # Bunch_ID
#         np.zeros(ParticleNum),  # Local_ID (will be set by the class)
#         np.zeros(ParticleNum)  # Global_ID (will be set by the class)
#     ))
#
#     # 创建一个 FFAG_Bunch 实例
#     bunch = FFAG_Bunch(ParticlesDistribution)
#
#     # 定义 StepDump 实例
#     step_dump = StepDump(step_interval=100, num_particles_to_dump=10, save_folder=f"./step_dump")
#
#     # 模拟粒子轨迹积分的过程，执行多个 step 更新粒子位置
#     num_steps = 2000  # 模拟 2000 步
#     for step in range(num_steps):
#         # 更新 Pre-steps (积分之前)
#         bunch.UpdatePreSteps()
#
#         # 模拟粒子轨迹积分 (简单模拟粒子方位角变化)
#         bunch.LocalBunch[:, 4] += 0.01  # 每步让粒子的方位角增加
#
#         # 更新 Post-steps (积分之后)
#         bunch.UpdatePostSteps()
#
#         # 每隔一定步数进行粒子数据的 Dump
#         step_dump.check_and_dump(step, bunch)
#
#     # print(f"Simulation completed on rank {rank}")
