SELECT * FROM name AS n1, role_type AS rt, cast_info AS ci, title AS t WHERE rt.role = 'costume designer' AND n1.id = ci.person_id AND ci.person_id = n1.id AND ci.movie_id = t.id AND t.id = ci.movie_id AND ci.role_id = rt.id AND rt.id = ci.role_id;