def extractParams(param_str):
    """
    Extracts parameters into a dict from string.
    Input:
        1. param_str (str) string of parameters in format “k1:10;k2:20” 

    Returns:
        2. mydict (dict) with format {k1: 10, k2:20}
    """
    #default parameters?
    if ',' in param_str:
        raise NameError(' , is an invalid character for parameter string.')
    if ':' not in param_str:
        raise NameError('No parameters found.')
    mydict = {}
    mystr = param_str.split(';')
    for s in mystr:
        k,v = s.split(':')
        mydict[k.strip()] = v.strip()
    return mydict

def formatSpecies(x, find, replace):
    """
    Processes strings to make compliant with Tellurium model. 
    Input:
        1. x (str) string to process
        2. Find (list of str) characters to remove
        3. Replace (list of str) characters to replace characters in “Find” with. Must have same length as Find.

    Returns:
        1. x (str) processed string
    """

    x = x.upper()

    try:
        if getStoich(x)[0][-1].isnumeric() and getStoich(x)[1][0] != ' ':
            x = 'z'+x
    except:
        pass

    for f,r in zip(find,replace):
        x = x.replace(f,r)
    x = x.strip()
    return x

FIND = ['-',',','+',' ']
REPLACE = ['_','_','_plus','']
fmt = lambda x: formatSpecies(x, FIND, REPLACE)

def getStoich(sp):
    i = 0
    if len(sp)>0:
        while sp[i].isnumeric():
            i+=1

    return sp[:i], sp[i:]