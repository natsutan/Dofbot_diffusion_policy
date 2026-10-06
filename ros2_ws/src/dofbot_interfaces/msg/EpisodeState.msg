std_msgs/Header header

uint8 PHASE_IDLE=0
uint8 PHASE_RESETTING=1
uint8 PHASE_READY=2
uint8 PHASE_RUNNING=3
uint8 PHASE_SUCCESS=4
uint8 PHASE_TERMINATED=5

uint32 episode_id
uint8 phase
int64 seed

float64 cube_x
float64 cube_y
float64 cube_z
float64 lift_height
float64 gripper_distance

bool success
bool terminated
bool truncated
string termination_reason
