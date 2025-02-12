# Running eval for Reasoning-gym datasets

### Install reasoning-gym
pip install reasoning-gym

### Config
// [
//     {
//       "name": "complex_arithmetic",
//       "size": 50,
//       "seed": 42
//     },
//     {
//       "name": "intermediate_arithmetic",
//       "size": 50,
//       "seed": 42
//     },
//     {
//       "name": "polynomial_equations",
//       "size": 50,
//       "seed": 42
//     },
//     {
//       "name": "polynomial_multiplication",
//       "size": 50,
//       "seed": 42
//     },
//     {
//       "name": "simple_equations",
//       "size": 50,
//       "seed": 42
//     },
//     {
//       "name": "simple_integration",
//       "size": 50,
//       "seed": 42
//     }
//   ]

# Model to use:
qwen/qwen-vl-plus:free

python eval.py --model qwen/qwen-vl-plus:free --config config.json

python eval.py --model anthropic/claude-3.5-sonnet --config config.json