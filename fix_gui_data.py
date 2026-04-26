import re

with open("x64/gemma3N_E4B/gui_app.py", "r") as f:
    code = f.read()

# Fix the variable 'data' which is undefined at init_model
code = re.sub(
    r'gem_main\.WEIGHT_MODE = data\.get\("weight_mode", "INT4"\)\.upper\(\)',
    'gem_main.WEIGHT_MODE = "INT4"',
    code
)

with open("x64/gemma3N_E4B/gui_app.py", "w") as f:
    f.write(code)
