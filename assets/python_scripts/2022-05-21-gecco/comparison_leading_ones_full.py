import numpy as np
from pyodide import create_once_callable
from js import console, requestAnimationFrame, clearTimeout, setTimeout  # clearInterval, setInterval (as alternatives)

# Get the HTML Input and output elements (Copy of HTML is show at the end of the file)
comparison_start_button = Element('start-btn-comparison-static')
comparison_reset_button = Element('reset-btn-comparison-static')
comparison_stop_button = Element('stop-btn-comparison-static')
comparison_speed_slider = Element('SpeedSlider-comparison-static')
comparison_bitflip_slider = Element('LambdaSlider-comparison-static')

comparison_canvas = Element('comparison-canvas-static')
comparison_ctx = comparison_canvas.element.getContext('2d')
comparison_canvas_optimal = Element('comparison-canvas-optimal')
comparison_ctx_optimal = comparison_canvas_optimal.element.getContext('2d')

# setup global variables and seed
comparison_ID = None
comparison_already_started = False
comparison_current_step = 0
comparison_total_steps = 0
comparison_current_step_optimal = 0
comparison_total_steps_optimal = 0
rng = np.random.RandomState(42)
rng_optimal = np.random.RandomState(42)
static_solves = 0
optimal_solves = 0

static_solve_steps = []
optimal_solve_steps = []

comparison_problem = rng.randint(2, size=(1, 15)).astype(bool)
comparison_problem_optimal = comparison_problem.copy()


def comparison_get_fitness(problem):
    """
    Counts the leading ones of the problem array
    """
    sum_ = 0
    for i in range(problem.shape[1]):
        sum_ += np.prod(problem[0, 0:i + 1])
    return sum_


def comparison_mutate(local_problem, num_bits, rng):
    """
    Performs one step of (1+1)RLS
    """
    new_problem = local_problem.copy()
    # Flip bits of randomly selected points
    bit_locs = rng.choice(local_problem.shape[1], size=num_bits, replace=False)
    new_problem[:, bit_locs] = ~new_problem[:, bit_locs]

    # keep result if fitness is better
    new_fit = comparison_get_fitness(new_problem)
    old_fit = comparison_get_fitness(local_problem)
    if new_fit >= old_fit:
        return new_problem
    else:
        return local_problem


def comparison_do_plot(ctx, canvas, problem, current_step, total_steps, static=True):
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
    fitness = comparison_get_fitness(problem)
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

    fitness = comparison_get_fitness(problem)

    if static:
        # Output Staticstics on page (first element is the id of the html element, second is the value)
        pyscript.write('plot-comparison-static', comparison_get_fitness(problem))
        # pyscript.write('stepsdiv', current_step)
        pyscript.write('timediv-comparison-static', total_steps)
    else:
        # Output Staticstics on page (first element is the id of the html element, second is the value)
        pyscript.write('plot-comparison-optimal', comparison_get_fitness(problem))
        # pyscript.write('stepsdiv', current_step)
        pyscript.write('timdiv-comparison-optimal', total_steps)

    return fitness, n_cells, current_step


def comparison_rls_one_step_and_plot(*args):
    """
    RLS with plotting
    """
    global comparison_ID, comparison_problem, comparison_already_started, comparison_current_step, comparison_total_steps
    global comparison_ctx, comparison_canvas, comparison_ctx_optimal, comparison_canvas_optimal, comparison_problem_optimal
    global comparison_current_step_optimal, comparison_total_steps_optimal, static_solves, optimal_solves, rng, rng_optimal
    global static_solve_steps, optimal_solve_steps

    # avoid starting the animation twice
    if comparison_already_started and args[0] != 0:
        return
    elif not comparison_already_started:
        comparison_ctx.clearRect(0, 0, comparison_canvas.element.width, comparison_canvas.element.height)  # clear comparison_canvas
    comparison_already_started = True

    num_bits = int(comparison_bitflip_slider.element.value)
    comparison_problem = comparison_mutate(comparison_problem, num_bits, rng=rng)
    comparison_current_step += num_bits
    fitness, n_cells, comparison_current_step = comparison_do_plot(
        comparison_ctx, comparison_canvas, comparison_problem, comparison_current_step, comparison_total_steps)

    optimal_fitness = comparison_get_fitness(comparison_problem_optimal)
    num_bits = n_cells//(optimal_fitness + 1)
    pyscript.write('r-comparison-optimal', str(num_bits))
    pyscript.write('opt-f-val', str(optimal_fitness))
    comparison_problem_optimal = comparison_mutate(comparison_problem_optimal, num_bits, rng=rng_optimal)
    comparison_current_step_optimal += num_bits
    optimal_fitness, n_cells, comparison_current_step = comparison_do_plot(
        comparison_ctx_optimal, comparison_canvas_optimal, comparison_problem_optimal,
        comparison_current_step_optimal, comparison_total_steps_optimal, static=False)

    if fitness != n_cells:
        comparison_total_steps += 1
    else:
        static_solve_steps.append(comparison_total_steps + 1)
        static_solves += 1
        rng = np.random.RandomState(static_solves)
        comparison_problem = rng.randint(2, size=comparison_problem.shape).astype(bool)
        comparison_current_step = 0
        comparison_total_steps = 0

        pyscript.write('avg-comparison-static', f'{np.mean(static_solve_steps):6.1f}')
        pyscript.write('solved-comparison-static', str(static_solves))

    if optimal_fitness != n_cells:
        comparison_total_steps_optimal += 1
        optimal_solve_steps.append(comparison_total_steps_optimal + 1)
    else:
        optimal_solves += 1
        rng_optimal = np.random.RandomState(optimal_solves)
        comparison_problem = rng.randint(2, size=comparison_problem.shape).astype(bool)
        comparison_current_step_optimal = 0
        comparison_total_steps_optimal = 0
        comparison_problem_optimal = comparison_problem.copy()

        pyscript.write('avg-comparison-optimal', f'{np.mean(optimal_solve_steps):6.1f}')
        pyscript.write('solved-comparison-optimal', str(optimal_solves))

    # This function will be called again after the specified time (i.e. 1000 / speed_slider.element.value)
    # ID is tracked so we can stop the animation on user request or when the solution is found
    comparison_ID = setTimeout(
        create_once_callable(comparison_rls_one_step_and_plot), 1000 / int(comparison_speed_slider.element.value), 0)


def comparison_clear_interval(*args):
    """
    Stop animation
    """
    global comparison_ID, comparison_already_started
    if comparison_already_started:
        clearTimeout(comparison_ID)
        comparison_ID = None
        comparison_already_started = False


def comparison_reset_solution(*args):
    """
    Reset the solution to the initial state
    """
    global comparison_problem, comparison_ID, comparison_current_step, comparison_total_steps
    global comparison_current_step_optimal, comparison_total_steps_optimal, comparison_problem_optimal
    global static_solves, static_solve_steps, optimal_solves, optimal_solve_steps
    comparison_problem = rng.randint(2, size=comparison_problem.shape).astype(bool)
    comparison_problem_optimal = comparison_problem.copy()
    comparison_current_step = 0
    comparison_total_steps = 0
    comparison_current_step_optimal = 0
    comparison_total_steps_optimal = 0

    static_solves = 0
    optimal_solves = 0

    static_solve_steps = []
    optimal_solve_steps = []

    pyscript.write('avg-comparison-static', str(0))
    pyscript.write('solved-comparison-static', str(0))
    pyscript.write('avg-comparison-optimal', str(0))
    pyscript.write('solved-comparison-optimal', str(0))

    comparison_clear_interval()
    comparison_ID = setTimeout(
        create_once_callable(comparison_rls_one_step_and_plot), 1000 / int(comparison_speed_slider.element.value), 0)


def comparison_respond_to_slider_change(*args):
    """
    Immediately react to speed change in animation
    """
    global comparison_ID, comparison_already_started
    if comparison_already_started:
        clearTimeout(comparison_ID)
        comparison_ID = None
        comparison_ID = setTimeout(
            create_once_callable(comparison_rls_one_step_and_plot),
            1000 / int(comparison_speed_slider.element.value), 0)
        comparison_speed_slider.element.nextElementSibling.value = comparison_speed_slider.element.value


# Add event listeners
comparison_start_button.element.onclick = comparison_rls_one_step_and_plot
comparison_stop_button.element.onclick = comparison_clear_interval
comparison_reset_button.element.onclick = comparison_reset_solution
comparison_speed_slider.element.oninput = comparison_respond_to_slider_change

# Render initial state
_ = comparison_do_plot(comparison_ctx, comparison_canvas, comparison_problem,
                       comparison_current_step, comparison_total_steps)
_ = comparison_do_plot(comparison_ctx_optimal, comparison_canvas_optimal, comparison_problem_optimal,
                       comparison_current_step_optimal, comparison_total_steps_optimal, static=False)
f = comparison_get_fitness(comparison_problem_optimal)
n = comparison_problem_optimal.shape[1] // (f + 1)
pyscript.write('r-comparison-optimal', str(n))
pyscript.write('opt-f-val', str(f))
