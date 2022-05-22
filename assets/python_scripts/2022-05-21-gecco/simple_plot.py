import numpy as np
from pyodide import create_once_callable
from js import console, requestAnimationFrame, clearTimeout, setTimeout  # clearInterval, setInterval (as alternatives)

# Get the HTML Input and output elements (Copy of HTML is show at the end of the file)
start_button = Element('start-btn')
reset_button = Element('reset-btn')
stop_button = Element('stop-btn')
speed_slider = Element('SpeedSlider')
bitflip_slider = Element('LambdaSlider')

canvas = Element('envcanvas')
ctx = canvas.element.getContext('2d')

# setup global variables and seed
ID = None
animation_started = False
current_step = 0
total_steps = 0
np.random.seed(42)

problem = np.random.randint(2, size=(1, 15)).astype(bool)


def get_fitness(problem):
    """
    Counts the leading ones of the problem array
    """
    sum_ = 0
    for i in range(problem.shape[1]):
        sum_ += np.prod(problem[0, 0:i + 1])
    return sum_


def do_plot(problem, current_step, total_steps):
    """
    Plotting function that plots the contents of the problem array
    :param problem:
    :param current_step:
    :param total_steps:
    :return:
    """
    n_cells = problem.shape[1]
    border_width = 2
    border_height = 2
    cell_width = (canvas.element.width / n_cells) - border_width
    cell_height = canvas.element.height - border_height
    fitness = get_fitness(problem)
    for s in range(n_cells):
        if s < fitness:
            ctx.fillStyle = 'lightgreen'
        else:
            ctx.fillStyle = 'white'
        position = [0, s]  # code could be used for multi rows

        # Color in the background
        ctx.fillRect(1. + position[1] * (cell_width + border_width), 1. + position[0] * (cell_height + border_height),
                     cell_width, cell_height)

        # Render the cell value
        ctx.font = '10px serif'
        ctx.fillStyle = 'black'
        value_str = f'{problem[0, s]:d}'
        ctx.fillText(value_str, 1. + position[1] * (cell_width + border_width) + cell_width / 2 - cell_width * .2,
                     1. + position[0] * (cell_height + cell_height) + cell_height / 2 + cell_height * .2)

    fitness = get_fitness(problem)

    # Output Staticstics on page (first element is the id of the html element, second is the value)
    pyscript.write('plot', get_fitness(problem))
    # pyscript.write('stepsdiv', current_step)
    pyscript.write('timediv', total_steps)

    return fitness, n_cells, current_step


def mutate(local_problem, num_bits):
    """
    Performs one step of (1+1)RLS
    """
    new_problem = local_problem.copy()
    # Flip bits of randomly selected points
    bit_locs = np.random.choice(local_problem.shape[1], size=num_bits, replace=False)
    new_problem[:, bit_locs] = ~new_problem[:, bit_locs]

    # keep result if fitness is better
    new_fit = get_fitness(new_problem)
    old_fit = get_fitness(local_problem)
    if new_fit >= old_fit:
        return new_problem
    else:
        return local_problem


def rls_one_step_and_plot(*args):
    """
    RLS with plotting
    """
    global ID, problem, animation_started, current_step, total_steps

    # avoid starting the animation twice
    if animation_started and args[0] != 0:
        return
    elif not animation_started:
        ctx.clearRect(0, 0, canvas.element.width, canvas.element.height)  # clear canvas
    animation_started = True

    num_bits = int(bitflip_slider.element.value)
    total_steps += 1
    problem = mutate(problem, num_bits)
    current_step += num_bits
    fitness, n_cells, current_step = do_plot(problem, current_step, total_steps)
    if fitness == n_cells:
        return clear_interval()
    else:
        # This function will be called again after the specified time (i.e. 1000 / speed_slider.element.value)
        # ID is tracked so we can stop the animation on user request or when the solution is found
        ID = setTimeout(create_once_callable(rls_one_step_and_plot), 1000 / int(speed_slider.element.value), 0)


def clear_interval(*args):
    """
    Stop animation
    """
    global ID, animation_started
    if animation_started:
        clearTimeout(ID)
        ID = None
        animation_started = False


def reset_solution(*args):
    """
    Reset the solution to the initial state
    """
    global problem, ID, current_step, total_steps
    problem = np.random.randint(2, size=problem.shape).astype(bool)
    current_step = 0
    total_steps = 0
    clear_interval()
    ID = setTimeout(create_once_callable(rls_one_step_and_plot), 1000 / int(speed_slider.element.value), 0)


def respond_to_slider_change(*args):
    """
    Immediately react to speed change in animation
    """
    global ID, animation_started
    if animation_started:
        clearTimeout(ID)
        ID = None
        ID = setTimeout(
            create_once_callable(rls_one_step_and_plot),
            1000 / int(speed_slider.element.value), 0)
        speed_slider.element.nextElementSibling.value = speed_slider.element.value


# Add event listeners
start_button.element.onclick = rls_one_step_and_plot
stop_button.element.onclick = clear_interval
reset_button.element.onclick = reset_solution
speed_slider.element.oninput = respond_to_slider_change

# Render initial state
_ = do_plot(problem, current_step, total_steps)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ COPY OF THE CUSTOM HTML FOR EMBEDDING THIS EXAMPLE CODE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ COPY OF THE CUSTOM HTML FOR EMBEDDING THIS EXAMPLE CODE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ COPY OF THE CUSTOM HTML FOR EMBEDDING THIS EXAMPLE CODE
# <div id="pyexample">
#     <py-script src="{{ '/assets/python_scripts/simple_plot.py' | relative_url }}"></py-script>
#     <canvas id="envcanvas" height="20" style="background-color: black; alignment: center; width: 100%">
#     Loading...
#     </canvas>
#     <div class="output-group">
#         <span class="output-group-btn">
#             <div style="display: inline-block">LeadingOnes: </div><div id="plot" style="display: inline-block;
#                   margin-right: 10px">0</div>
#             <!--<div style="display: inline-block">Mutations Tried: </div><div id="stepsdiv" style="
#                   display: inline-block">0</div>-->
#             <div style="display: inline-block">Steps Taken: </div><div id="timediv" style="
#                   display: inline-block">0</div>
#         </span>
#     </div>
#     <div class="input-group">
#         <span class="input-group-btn">
#             <button type="button" class="btn btn-default" id="start-btn" style="alignment: left">START</button>
#             <button type="button" class="btn btn-default" id="reset-btn" style="alignment: center">RESET</button>
#             <button type="button" class="btn btn-default" id="stop-btn" style="alignment: right">STOP</button>
#             <input type="range" min="1" max="100" value="1" step="1" class="slider" id="SpeedSlider" style="
#                 -webkit-appearance: none;
#                 width: 21.5%;
#                 height: 10px;
#                 background: #d3d3d3;
#                 outline: none;
#                 opacity: 0.7;
#                 -webkit-transition: .2s;
#                 transition: opacity .2s;
#                 display: inline-block;
#                 margin-right: 10px;
#             " oninput="this.nextElementSibling.value = this.value">
#             Speed: <output>1</output>%
#             <input type="range" min="1" max="10" value="1" step="1" class="slider" id="LambdaSlider" style="
#                 -webkit-appearance: none;
#                 width: 73%;
#                 height: 10px;
#                 background: #d3d3d3;
#                 outline: none;
#                 opacity: 0.7;
#                 -webkit-transition: .2s;
#                 transition: opacity .2s;
#                 display: inline-block;
#                 margin-right: 10px;
#             " oninput="this.nextElementSibling.value = this.value">
#             Number of bitflips: <output>1</output>
#         </span>
#     </div>
# </div>
