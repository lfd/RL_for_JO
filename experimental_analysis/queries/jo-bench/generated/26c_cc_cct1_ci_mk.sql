SELECT * FROM cast_info AS ci, movie_keyword AS mk, complete_cast AS cc, comp_cast_type AS cct1 WHERE cct1.kind = 'cast' AND mk.movie_id = ci.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id;