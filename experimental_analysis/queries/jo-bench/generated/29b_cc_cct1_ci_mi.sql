SELECT * FROM comp_cast_type AS cct1, movie_info AS mi, complete_cast AS cc, cast_info AS ci WHERE cct1.kind = 'cast' AND ci.note IN ('(voice)', '(voice) (uncredited)', '(voice: English version)') AND mi.info LIKE 'USA:%200%' AND mi.movie_id = ci.movie_id AND ci.movie_id = mi.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id;