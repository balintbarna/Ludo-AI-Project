start_pos = 0
entered_pos = 1
goal_pos = 59
dice_enter_game = [6]
star_boost_seven = [5, 18, 31, 44]
star_boost_six = [12, 25, 38]
safe_globes = [1, 9, 14, 22, 27, 35, 40, 48, 53]
star_boost_eight = 51
safe_spots = []
quarter_map = 13
character_pos_offset = quarter_map
half_map = 2*quarter_map
full_map = 2*half_map
enemy_piece_max_reachable_pos = 53
max_score_divider = 4.0*goal_pos

def piece_destination(piece_pos, dice):
    if is_start(piece_pos):
        if dice in dice_enter_game:
            return entered_pos
        else:
            return start_pos

    dest = piece_pos + dice
    if dice > goal_pos:
        dest = goal_pos - dice
    elif dest in star_boost_six:
        dest += 6
    elif dest in star_boost_seven:
        dest += 7
    elif dest is star_boost_eight:
        dest += 8
    return dest

def is_goal(pos):
    return pos is goal_pos

def is_start(pos):
    return pos is start_pos

def can_enter_game(pos, dice):
    return dice in dice_enter_game and is_start(pos)

def are_pieces_at_pos(pos, pieces_list):
    return pos in pieces_list

def can_build_tower(dest, own_pieces_list):
    return are_pieces_at_pos(dest, own_pieces_list)

def transform_enemy_piece(own_piece_pos, enemy_piece_pos, list_index):
    if enemy_piece_pos > enemy_piece_max_reachable_pos: # in the finish line, cannot interact with other players
        return 1000
    
    offset = list_index * character_pos_offset
    offseted_pos = enemy_piece_pos + offset
    if offseted_pos > enemy_piece_max_reachable_pos: # unreachable distance
        offseted_pos -= full_map
    while (offseted_pos - own_piece_pos) > half_map: # distances larger than a half map are hardly relevant
        offseted_pos -= full_map

def is_safe(pos):
    return pos in safe_globes

def can_kill_enemy(dest, enemy_pieces_lists):
    for i in range(0, len(enemy_pieces_lists)):
        enemy_pieces_list = enemy_pieces_lists[i]
        for enemy_piece in enemy_pieces_list:
            transformed_enemy = transform_enemy_piece(dest, enemy_piece, i)
            if dest is transformed_enemy:
                if enemy_piece not in safe_globes or dest is entered_pos:
                    return True
    return False

def potential_player_progress(own_pieces, piece_index, dice):
    summer = 0
    for i in range(0, len(own_pieces)):
        if piece_index is i:
            summer += piece_destination(own_pieces[i], dice)
        else:
            summer += own_pieces[i]
    score = summer / max_score_divider
    return score # 0-1 point; 0: all in base; 1: all in goal (win)
