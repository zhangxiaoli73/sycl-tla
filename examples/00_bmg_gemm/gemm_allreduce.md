
让我们一起讨论gemm allreduce fuse在1个kernel里面的实现。


1) 申请一块大的symmetric memory，size 是 data 和 flag之和，前面部分是symm_data, 后面的symm_flag.
- Data 部分: `gemm_output_size`（本 rank 的 output）
- Flag 部分: `num_of_tiles * world_size`（每个 rank 的每个 tile 一个 flag，由 remote 写入通知）

2）分配出一部分workgroup专门用来做allreduce，可以叫做wg_ar，剩下的做gemm, 支持用环境变量来调整wg_ar的大小。

3) 每个gemm tile计算完之后，update 自己的symm_data, 同时也update remote symm_flag, 告诉remote机器我本地tile计算结果完成了可以访问。注意 fence 的位置：
- **Producer 侧**：先写 symm_data，然后 `sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system)`，最后再写 remote symm_flag。Release fence 必须在 data write 和 flag write **之间**，保证 data 先于 flag 对 remote 可见。
- **Consumer 侧**（wg_ar）：先 spin check 自己的 symm_flag，发现 flag 被 set 之后，`sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system)`，然后再去 pull remote 的 symm_data。Acquire fence 保证后续读到的 data 是 flag set 之后的最新值。

4）wg_ar里面load自己的symm_data, check remote rank的flag 有没有被update，如果某个tile在所有的rank上面都被告知已经计算完成，那么就pull remote的数据到本地，做sum操作，并把sum之后的结果更新到output对应的tile上面。

5) wg_ar 需要知道何时所有 tile 都已完成 reduce，否则会无限 spin。需要一个全局 atomic counter 记录已完成 reduce 的 tile 数，当 `completed_tiles == total_tiles` 时所有 wg_ar 退出。

7) wg_ar 内部的 subgroup 级分工

一个 wg_ar 内部切成多个 subgroup，每个 subgroup 负责 check/reduce 一部分 tile：
- **静态分配**：`tile_id % num_subgroups == subgroup_id` 的 tile 归该 subgroup 负责
- 同一 WG 内 subgroup 之间**不需要 atomic 协调**，因为 tile 归属在 launch 时就确定了
- 每个 subgroup 内部的 work-item 协作完成一个 tile 的数据搬运（利用 subgroup block load/store）
- subgroup 内逻辑：spin check 自己负责的 tile flags → 全部 rank ready 后 pull remote data → local sum → 写回 output

跨 wg_ar 之间仍然需要协调（不同 wg_ar 负责不同 tile range），可以用：
- 静态划分：`tile_id / tiles_per_wg` 决定哪个 wg_ar 负责
- 或者 atomic claim：per-tile bitmap，wg_ar 用 `atomic_compare_exchange` 独占

8）本算法只支持m在1024以内的大小。

