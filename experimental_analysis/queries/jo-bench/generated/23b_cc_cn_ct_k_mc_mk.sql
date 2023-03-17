SELECT * FROM company_name AS cn, company_type AS ct, keyword AS k, complete_cast AS cc, movie_keyword AS mk, movie_companies AS mc WHERE cn.country_code = '[us]' AND k.keyword IN ('nerd', 'loner', 'alienation', 'dignity') AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND cn.id = mc.company_id AND mc.company_id = cn.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id;