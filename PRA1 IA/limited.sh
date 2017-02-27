#!/usr/bin/env sh

cutoff=""
memlim=""


# Parsing command line
#------------------------------------------------------------------------------

if [ $# -lt 1 ]; then
    echo "Usage: $0 [-t time] [-m memory] <command> <parameters>"
    echo "    * time: CPU time in seconds"
    echo "    * memory: Virtual memory in bytes"
    exit 1
fi

while [ "$#" -gt 0 ]; do
    key="$1"
    case $key in
        -t|--time)
            cutoff="$2"
            shift
        ;;
        -m|--memory)
            memlim="$2"
            shift
        ;;
        *)
            break
            #echo "----"
            #echo "Unknown option: $key"
            #echo "----"
            #exit 2
        ;;
    esac

    shift
done


# Check if there are still arguments (command and parameters)
#------------------------------------------------------------------------------

if [ $# -lt 1 ]; then
    echo "Error: nothing to be run"
    exit 2
fi


# Setting limits
#------------------------------------------------------------------------------

#echo "memlim=$memlim"
#echo "cutoff=$cutoff"

if [[ ! -z $memlim &&  ! -z $cutoff ]]; then
    echo "Use -t or -m to specify a limit"
    exit 5
fi


if [ ! -z $memlim ]; then
    ulimit -v $memlim
    if [ $? -ne 0 ]; then
        echo "Error setting memory limit"
        exit 3
    fi
fi

if [ ! -z $cutoff ]; then
    ulimit -t $cutoff
    if [ $? -ne 0 ]; then
        echo "Error setting time limit"
        exit 4
    fi
fi


# Execute command
#------------------------------------------------------------------------------
echo "$@"
time -p "$@"
