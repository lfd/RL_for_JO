SELECT * FROM keyword AS k, role_type AS rt, cast_info AS ci, aka_name AS an, movie_keyword AS mk WHERE ci.note IN ('(voice)', '(voice) (uncredited)', '(voice: English version)') AND k.keyword = 'computer-animation' AND rt.role = 'actress' AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND rt.id = ci.role_id AND ci.role_id = rt.id AND ci.person_id = an.person_id AND an.person_id = ci.person_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;