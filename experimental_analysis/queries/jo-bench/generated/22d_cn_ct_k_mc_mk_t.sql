SELECT * FROM movie_keyword AS mk, keyword AS k, company_type AS ct, company_name AS cn, movie_companies AS mc, title AS t WHERE cn.country_code != '[us]' AND k.keyword IN ('murder', 'murder-in-title', 'blood', 'violence') AND t.production_year > 2005 AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id AND cn.id = mc.company_id AND mc.company_id = cn.id;