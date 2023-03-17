SELECT * FROM cast_info AS ci, role_type AS rt, movie_keyword AS mk, aka_name AS an, title AS t WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND rt.role = 'actress' AND t.production_year BETWEEN 2000 AND 2010 AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND rt.id = ci.role_id AND ci.role_id = rt.id AND ci.person_id = an.person_id AND an.person_id = ci.person_id;