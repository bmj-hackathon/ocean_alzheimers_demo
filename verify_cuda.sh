autoload colors 
if [[ "$terminfo[colors]" -gt 8 ]]; then 
    colors 
fi 
for COLOR in RED GREEN YELLOW BLUE MAGENTA CYAN BLACK WHITE; do 
    eval $COLOR='$fg_no_bold[${(L)COLOR}]' 
    eval BOLD_$COLOR='$fg_bold[${(L)COLOR}]' 
done 
eval RESET='$reset_color' 

#############################
PYTHON_CODE=$(cat <<END
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

get_available_gpus()
END
)

#############################

pipenv shell

#############################

echo ""
echo "${BOLD_CYAN}Linux version${NOCOLOR}"
uname -m 
uname -a
cat /etc/upstream-release/lsb-release
echo ""

echo "${BOLD_CYAN}GCC version${NOCOLOR}"
gcc --version

echo "${BOLD_CYAN}Video card version${NOCOLOR}"
lspci | grep -i nvidia
echo ""

echo "${BOLD_CYAN}GPU nvcc version${NOCOLOR}"
which nvcc
nvcc --version
echo ""

echo "${BOLD_CYAN}Dynamic linker run-time bindings${NOCOLOR}"
ldconfig -p | grep cuda
echo ""

echo "${BOLD_CYAN}Python Version${NOCOLOR}"
python --version
echo ""
 
echo "${BOLD_CYAN}Library Path for CUDA${NOCOLOR}"
echo $LD_LIBRARY_PATH  
echo ""

#############################
echo "${BOLD_CYAN}Check python import and devices${NOCOLOR}"
res="$(python3 -c "$PYTHON_CODE")"

# continue with bash code
echo "$res"
