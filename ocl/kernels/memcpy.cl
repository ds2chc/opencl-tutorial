/// Get index of matrix.
/// Note that this function is based on row-major
#define get_2d_index(i, j, num_rows, num_cols) ((i) * (num_cols) + (j))

/// num_threads := 4
/// blocks: {num_threads, num_threads}
/// groups: {row_size/blocks[0], col_size/blocks[1]}
__kernel void copy_matrix(int row_size, int col_size, 
                          const __global float* src,
                          __global float* tgt) {
  int row_offset = get_group_id(0) * get_local_size(0);
  int col_offset = get_group_id(1) * get_local_size(1);

  int i = get_local_id(0) + get_local_size(1);
  int j = get_local_id(1);  
  
  if (row_offset + i < row_size && col_offset + j < col_size) {
    int index = get_2d_index(
      row_offset + i, col_offset + j, row_size, col_size);
    tgt[index] = src[index];
  }
}
