def try_all(fn, list, E=Exception):
    for i in list:
        try:
            return fn(i)
        except E as e:
            continue
