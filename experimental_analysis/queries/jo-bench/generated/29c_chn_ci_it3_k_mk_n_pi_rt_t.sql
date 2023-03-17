SELECT * FROM info_type AS it3, role_type AS rt, name AS n, title AS t, keyword AS k, movie_keyword AS mk, cast_info AS ci, person_info AS pi, char_name AS chn WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND it3.info = 'trivia' AND k.keyword = 'computer-animation' AND n.gender = 'f' AND n.name LIKE '%An%' AND rt.role = 'actress' AND t.production_year BETWEEN 2000 AND 2010 AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND n.id = ci.person_id AND ci.person_id = n.id AND rt.id = ci.role_id AND ci.role_id = rt.id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND n.id = pi.person_id AND pi.person_id = n.id AND ci.person_id = pi.person_id AND pi.person_id = ci.person_id AND it3.id = pi.info_type_id AND pi.info_type_id = it3.id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;