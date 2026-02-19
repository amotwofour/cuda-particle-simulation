# CUDA C++ Project

This project is a simple CUDA C++ application that demonstrates the use of CUDA for parallel computing. It includes an entry point and utility functions that can be used for various computations.

## Project Structure

```
cuda-cpp-project
├── src
│   ├── main.cu       # Entry point of the application
│   └── utils.cu      # Utility functions and CUDA kernels
├── include
│   └── utils.h       # Header file for utility functions
├── CMakeLists.txt    # CMake configuration file
└── README.md         # Project documentation
```

## Requirements

- CUDA Toolkit
- CMake

## Building the Project

1. Clone the repository or download the project files.
2. Open a terminal and navigate to the project directory.
3. Create a build directory:
   ```
   mkdir build
   cd build
   ```
4. Run CMake to configure the project:
   ```
   cmake ..
   ```
5. Build the project:
   ```
   make
   ```

## Running the Application

After building the project, you can run the application using the following command:
```
./cuda-cpp-project
```

## Usage

This project serves as a template for developing CUDA applications. You can modify the utility functions in `utils.cu` and declare new functions in `utils.h` as needed. The `main.cu` file is where you can implement your application logic and kernel launches.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.