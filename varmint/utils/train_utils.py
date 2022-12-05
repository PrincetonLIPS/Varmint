def update_ewa(ewa_loss, loss, ewa_weight=0.95):
    if ewa_loss == None:
        ewa_loss = loss
    else:
        ewa_loss = ewa_loss * ewa_weight + loss * (1 - ewa_weight)

    return ewa_loss
