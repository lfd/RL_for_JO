SELECT * FROM name AS n, cast_info AS ci, movie_companies AS mc, title AS t, movie_keyword AS mk, keyword AS k, char_name AS chn, company_name AS cn, aka_name AS an WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND cn.country_code = '[us]' AND k.keyword IN ('hero', 'martial-arts', 'hand-to-hand-combat') AND n.gender = 'f' AND n.name LIKE '%An%' AND t.production_year > 2010 AND t.id = mc.movie_id AND mc.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND mc.movie_id = ci.movie_id AND ci.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id AND n.id = ci.person_id AND ci.person_id = n.id AND n.id = an.person_id AND an.person_id = n.id AND ci.person_id = an.person_id AND an.person_id = ci.person_id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;