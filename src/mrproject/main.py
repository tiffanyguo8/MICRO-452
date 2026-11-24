from navigation.py import nav

def test() -> None:
    print("The test function works")

def main() -> None:
    state = "INIT"

    while True:
        if state == "INIT":
            #get map, start, goal
            state = "PLAN"

        elif state == "PLAN":
            #plan path with A*
            #smooth path
            #transfer waypoints
            path_found = nav(Map, SearchStart, SearchGoal, "Octile")
            if path_found:
                state = "MOVE"
            else:
                state = "ERROR" 

        elif state == "MOVE":
            #follow path
            #get the sensors mes

            if goal_reached:
                state = "DONE"
            elif obstacle_detected:
                state = "OBST_DETECTED"
    
        elif state == "OBST_DETECTED":
            #eval if path can still followable
            if path_still_valid:
                state = "MOVE"
            else:
                state = "REPLAN"
    
        elif state == "REPLAN":
            #act map
            # get act pos°
            # replan path with A*

            state = "PLAN"
    
        elif state == "DONE":
            #stop
            break

        elif state == "ERROR":
            #stop and signal pbm
            break