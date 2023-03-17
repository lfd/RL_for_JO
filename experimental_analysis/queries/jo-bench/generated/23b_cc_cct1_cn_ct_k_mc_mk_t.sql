SELECT * FROM company_name AS cn, title AS t, movie_companies AS mc, keyword AS k, complete_cast AS cc, comp_cast_type AS cct1, movie_keyword AS mk, company_type AS ct WHERE cct1.kind = 'complete+verified' AND cn.country_code = '[us]' AND k.keyword IN ('nerd', 'loner', 'alienation', 'dignity') AND t.production_year > 2000 AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND cn.id = mc.company_id AND mc.company_id = cn.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id AND cct1.id = cc.status_id AND cc.status_id = cct1.id;