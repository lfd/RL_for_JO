SELECT * FROM person_info AS pi, movie_keyword AS mk, complete_cast AS cc, cast_info AS ci, role_type AS rt WHERE ci.note IN ('(voice)', '(voice) (uncredited)', '(voice: English version)') AND rt.role = 'actress' AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND rt.id = ci.role_id AND ci.role_id = rt.id AND ci.person_id = pi.person_id AND pi.person_id = ci.person_id;