#!/n/fs/mm-iga/miniconda3/envs/igaconda38/bin/python

"""
Example use:
    ./launch_batch_experiments.py experiment.txt --experiment-name=test_experiment --experiment-dir=test_exp_dir --slurm-overrides="mem=31G,gres=gpu:1"

"""

import re
import os
import stat
import shutil
import argparse
import random
import itertools


slurm_param_dict = {
    'mem': '32G',
    'time': '34:00:00',
    'gres': 'gpu:1',
    'job_name': 'untitled',
    'output': 'output.txt',
}


TEMPLATE = \
    '''#!/bin/sh

#SBATCH --job-name={job_name}
#SBATCH --output={output}
#SBATCH --account=lips
#
#SBATCH --gres={gres}
#SBATCH --mem={mem}
#SBATCH --time={time}
#

'''


def make_exec(filename):
    st = os.stat(filename)
    os.chmod(filename, st.st_mode | 0o111)


def substitute_command_parameters(command, substitutions):
    for sub in substitutions.keys():
        command = command.replace('{'+sub+'}', substitutions[sub])
    return command


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='SLURM Parameter Sweeper')
    parser.add_argument(
        'filename', help='python file (relative or absolute) to run')
    parser.add_argument('--experiment-name', help='experiment name')
    parser.add_argument('--slurm-overrides',
                        help='changes from slurm defaults')
    parser.add_argument(
        '--experiment-dir', default='/n/fs/mm-iga/Varmint/', help='experiment base directory')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite the existing experiment directory.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show full command to run for each worker.')
    args = parser.parse_args()
    print("Using experiment file {}".format(args.filename))

    if not args.experiment_name:
        args.experiment_name = 'exp{}'.format(random.randint(100000, 999999))

    base_dir = os.path.join(args.experiment_dir, 'slurm_experiments')
    if not os.path.isdir(base_dir):
        os.mkdir(base_dir)

    # Create experiment directory
    exp_dir = os.path.join(base_dir, '{}'.format(args.experiment_name))
    slurm_output_dir = os.path.join(
        base_dir, '{}'.format(args.experiment_name), 'slurm_output')
    print("Creating experiment {} in {}".format(args.experiment_name, exp_dir))
    if os.path.isdir(exp_dir) and args.overwrite:
        print("Overwriting existing directory.")
        shutil.rmtree(exp_dir)
    elif os.path.isdir(exp_dir):
        print("Experiment {} exists. Aborting.".format(args.experiment_name))
        return

    os.mkdir(exp_dir)
    os.mkdir(slurm_output_dir)
    shutil.copyfile(args.filename, os.path.join(
        exp_dir, 'experiment_commands.txt'))

    # Parse slurm arguments
    if args.slurm_overrides:
        slurm_dict = args.slurm_overrides.split(',')
        for el in slurm_dict:
            (key, value) = el.split('=')
            if key not in slurm_param_dict:
                print("{} SLURM argument not supported. Skipping.".format(key))
            else:
                slurm_param_dict[key] = value

    # Parse parameter arguments
    exps = []
    with open(args.filename, 'r') as exp_file:
        exps = exp_file.readlines()

    # Use itertools to iterate over hyperparameter sweep.
    # TODO(doktay) Add more features.
    files_list = []
    i = 0
    pattern = re.compile(
        "( \-\-| \-)([a-zA-Z0-9\-\_]+)(=?[ ]*){P:([^\{]*):P\}")
    for exp_pre in exps:
        # If we have parameters of the form {P:1,2,3:P} expand to three different experiments.
        iterators = {}
        match = pattern.findall(exp_pre)
        for prefix, param, space, values in match:
            print(space)
            combinations = ['{}{}{}{}'.format(
                prefix, param, space, val) for val in values.split(',')]
            iterators[param] = combinations

        if len(iterators) > 0:
            keys, values = zip(*iterators.items())

            for param_combination in itertools.product(*values):
                # Substitute the parameter values for this combination.
                pdict = dict(zip(keys, param_combination))
                exp = pattern.sub(lambda m: pdict[m.groups()[1]], exp_pre)

                slurm_dict_copy = slurm_param_dict.copy()
                slurm_dict_copy['output'] = \
                    os.path.join(slurm_output_dir,
                                 'wid{0:04}output.txt'.format(i))
                slurm_dict_copy['job_name'] = str(
                    args.experiment_name) + 'wid{0:04}'.format(i)
                slurm_out = TEMPLATE.format(**slurm_dict_copy)
                command = 'srun ' + exp
                command = substitute_command_parameters(
                    command, {'wid': '{0:04}'.format(i), 'expdir': exp_dir})

                slurm_out = slurm_out + command + '\n'
                worker_filename = os.path.join(
                    exp_dir, 'wid{0:04}exp.sh'.format(i))
                with open(worker_filename, 'w') as f:
                    f.write(slurm_out)
                make_exec(worker_filename)
                files_list.append(worker_filename)
                print('Prepared worker {} at {}'.format(i, worker_filename))
                if args.verbose:
                    print('Will run the command: \n\t{}'.format(command))
                i += 1
        else:
            exp = exp_pre
            slurm_dict_copy = slurm_param_dict.copy()
            slurm_dict_copy['output'] = \
                os.path.join(slurm_output_dir, 'wid{0:04}output.txt'.format(i))
            slurm_dict_copy['job_name'] = str(
                args.experiment_name) + 'wid{0:04}'.format(i)
            slurm_out = TEMPLATE.format(**slurm_dict_copy)
            command = 'srun ' + exp
            command = substitute_command_parameters(
                command, {'wid': '{0:04}'.format(i), 'expdir': exp_dir})

            slurm_out = slurm_out + command + '\n'
            worker_filename = os.path.join(
                exp_dir, 'wid{0:04}exp.sh'.format(i))
            with open(worker_filename, 'w') as f:
                f.write(slurm_out)
            make_exec(worker_filename)
            files_list.append(worker_filename)
            print('Prepared worker {} at {}'.format(i, worker_filename))
            if args.verbose:
                print('Will run the command: \n\t{}'.format(command))
            i += 1

    print('Should I submit batch jobs? Y/N')
    while True:
        a = input()
        if a == 'Y':
            print('Submitting')
            for file in files_list:
                print('sbatch {}'.format(file))
                os.system('sbatch {}'.format(file))
            break
        elif a == 'N':
            print('Ok bye.')
            break
        else:
            print('Y or N')


if __name__ == '__main__':
    main()
