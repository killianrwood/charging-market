

def interval_projection(x,a,b): 
    return a*(x<a) + x*(a<x)*(x<b) + b*(x>b)

def primal_dual_projection(x,y,a,b,c,d):
    
    return (
        interval_projection(x,a,b),
        interval_projection(y,c,d)
    )
    



