SELECT * FROM movie_companies AS mc, cast_info AS ci, role_type AS rt, title AS t WHERE rt.role = 'writer' AND ci.movie_id = t.id AND t.id = ci.movie_id AND t.id = mc.movie_id AND mc.movie_id = t.id AND ci.role_id = rt.id AND rt.id = ci.role_id AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id;