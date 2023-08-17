/// Get index of matrix.
/// Note that this function is based on row-major
//#define get_2d_index(i, j, num_rows, num_cols) ((i) * (num_cols) + (j))
/// Note that this function is based on col-major
#define get_2d_index(i, j, num_rows, num_cols) ((i) + (j) * (num_rows))

#define TILE_SIZE 2

__kernel void matmul_v2(const int M, const int N, const int K,
                        const __global float* lhs,
                        const __global float* rhs,
                        __global float* results) {
  
  // Identify threads
  const int row = get_local_id(0);
  const int col = get_local_id(1);
  const int global_row = TILE_SIZE * get_group_id(0) + row; // row id of results, (0 ... M)
  const int global_col = TILE_SIZE * get_group_id(1) + col; // col id of results, (0 ... N)

  // Prepare local memory to fit a tile of TS*TS elements of A and B
  __local float local_lhs[TILE_SIZE][TILE_SIZE];
  __local float local_rhs[TILE_SIZE][TILE_SIZE];

  float acc = 0.0f;
  const int num_tiles = K / TILE_SIZE;
  for (int t = 0; t < num_tiles; t++) {
    // Load ont tile of lhs & rhs into local memory
    const int tile_row = TILE_SIZE * t + row;
    const int tile_col = TILE_SIZE * t + col;
    local_lhs[col][row] = lhs[tile_col * M + global_row];
    local_rhs[col][row] = rhs[global_col * K + tile_row];
    barrier(CLK_LOCAL_MEM_FENCE);

    // perform the computation for a single tile
    for (int k = 0; k < TILE_SIZE; k++) {
      acc += local_lhs[k][row] * local_rhs[col][k];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  int result_index = get_2d_index(global_row, global_col, M, N);
  result[result_index] = acc;
} 