
class LoopClosure():
    """A class to store all the info needed for a loop closure
    """
    
    def __init__(self,source_key,target_key,source_points,target_points,source_context,target_context) -> None:
        self.source_key = source_key
        self.target_key = target_key
        self.source_points = source_points
        self.target_points = target_points
        self.status = True
        self.message = "initialized"
        self.gi_transform = None
        self.icp_transform = None
        self.source_context = source_context
        self.target_context = target_context

        self.source_points_init = None
        self.reg_points = None 
        self.fit_score = None

        self.count = None
        self.ratio = None
        self.context_diff = None
        self.overlap = None