SELECT * FROM name AS n, title AS t, keyword AS k, complete_cast AS cc, movie_keyword AS mk, comp_cast_type AS cct2, role_type AS rt, cast_info AS ci, info_type AS it, movie_info AS mi WHERE cct2.kind = 'complete+verified' AND ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND it.info = 'release dates' AND k.keyword = 'computer-animation' AND mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%200%' OR mi.info LIKE 'USA:%200%') AND n.gender = 'f' AND n.name LIKE '%An%' AND rt.role = 'actress' AND t.production_year BETWEEN 2000 AND 2010 AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND mi.movie_id = ci.movie_id AND ci.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mi.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND it.id = mi.info_type_id AND mi.info_type_id = it.id AND n.id = ci.person_id AND ci.person_id = n.id AND rt.id = ci.role_id AND ci.role_id = rt.id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;