SELECT * FROM role_type AS rt, name AS n, title AS t, keyword AS k, complete_cast AS cc, movie_keyword AS mk, cast_info AS ci, person_info AS pi, aka_name AS an, info_type AS it3 WHERE ci.note IN ('(voice)', '(voice) (uncredited)', '(voice: English version)') AND it3.info = 'height' AND k.keyword = 'computer-animation' AND n.gender = 'f' AND n.name LIKE '%An%' AND rt.role = 'actress' AND t.title = 'Shrek 2' AND t.production_year BETWEEN 2000 AND 2005 AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND n.id = ci.person_id AND ci.person_id = n.id AND rt.id = ci.role_id AND ci.role_id = rt.id AND n.id = an.person_id AND an.person_id = n.id AND ci.person_id = an.person_id AND an.person_id = ci.person_id AND n.id = pi.person_id AND pi.person_id = n.id AND ci.person_id = pi.person_id AND pi.person_id = ci.person_id AND it3.id = pi.info_type_id AND pi.info_type_id = it3.id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;