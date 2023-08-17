#pragma once

#include <vector>


class Matrix {
public:
  Matrix() : row_(0), col_(0) {}

  Matrix(int rows, int cols) : row_(rows), col_(cols) {
    data_.resize(row_ * col_);
  }

  ~Matrix() {}

  int Rows() const { return row_; }
  int Cols() const { return col_; }
  int NumElmts() const { return row_ * col_; }

  float& operator() (int i, int j) {
    int index = GetFlattenedIndex(i, j);
    return data_[index];
  }

  const float operator() (int i, int j) const {
    int index = GetFlattenedIndex(i, j);
    return data_[index];
  }

  float* RawPtr() { return data_.data(); }
  const float* RawPtr() const { return data_.data(); }

private:
  int row_;
  int col_;
  std::vector<float> data_;

  int GetFlattenedIndex(int r, int c) const {
    //return r * col_ + c;
    return r + c * row_;
  }
}; // class Matrix